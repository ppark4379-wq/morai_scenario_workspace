#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import numpy as np
import pygame
import rospy
from morai_msgs.msg import EgoVehicleStatus

current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(current_path, 'lib')))

from class_defs import MGeo

# ── 설정 ──────────────────────────────────────────────
MAP_NAME    = 'R_KR_PG_KATRI'
SCREEN_W    = 1200
SCREEN_H    = 800
FPS         = 60

COLOR_BG     = (30,  30,  30)
COLOR_LINK   = (80,  80,  80)
COLOR_NODE   = (60, 120, 200)
COLOR_PATH   = (255, 200,   0)
COLOR_START  = (0,  220,   0)
COLOR_END    = (220,   0,   0)
COLOR_VEHICLE= (255, 100,   0)
COLOR_TEXT   = (220, 220, 220)
# ──────────────────────────────────────────────────────

# 차량 위치 공유 (ROS 콜백 ↔ pygame 메인루프)
vehicle_pos = {'x': None, 'y': None, 'yaw': 0.0}
vehicle_lock = threading.Lock()


def ego_callback(data):
    with vehicle_lock:
        vehicle_pos['x']   = data.position.x
        vehicle_pos['y']   = data.position.y
        vehicle_pos['yaw'] = np.deg2rad(data.heading)


def world_to_screen(points_xy, offset, scale):
    pts = np.array(points_xy, dtype=np.float64)
    sx = (pts[:, 0] - offset[0]) * scale + SCREEN_W / 2
    sy = -(pts[:, 1] - offset[1]) * scale + SCREEN_H / 2
    return np.stack([sx, sy], axis=1).astype(int)


def load_path_txt(filepath):
    pts = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            x, y = map(float, line.split())
            pts.append((x, y))
    return pts


def draw_vehicle(screen, sx, sy, yaw, scale):
    """차량을 삼각형 화살표로 그리기"""
    size = max(8, int(scale * 2))
    pts_local = np.array([
        [ size,      0],
        [-size,  size//2],
        [-size, -size//2],
    ], dtype=np.float64)

    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    pts_rotated = (rot @ pts_local.T).T
    # y축 반전 (화면 좌표계)
    pts_screen = pts_rotated * np.array([1, -1]) + np.array([sx, sy])
    pygame.draw.polygon(screen, COLOR_VEHICLE, pts_screen.astype(int).tolist())


def main():
    # ── ROS 초기화 ──
    rospy.init_node('mgeo_visualization', anonymous=True)
    rospy.Subscriber('/Ego_topic', EgoVehicleStatus, ego_callback)
    ros_thread = threading.Thread(target=rospy.spin, daemon=True)
    ros_thread.start()

    # ── 지도 로드 ──
    load_path = os.path.normpath(os.path.join(current_path, '../../../../' + MAP_NAME))
    print('지도 로드 중...')
    mgeo = MGeo.create_instance_from_json(load_path)
    nodes = mgeo.node_set.nodes
    links  = mgeo.link_set.lines
    print(f'노드: {len(nodes)}개  링크: {len(links)}개')

    # ── 링크/노드 전처리 ──
    link_polylines = []
    all_link_pts   = []
    for link in links.values():
        pts = [(p[0], p[1]) for p in link.points]
        link_polylines.append(pts)
        all_link_pts.extend(pts)

    node_pts = [(n.point[0], n.point[1]) for n in nodes.values()]
    all_arr  = np.array(all_link_pts + node_pts)
    cx, cy   = all_arr[:, 0].mean(), all_arr[:, 1].mean()
    span     = max(all_arr[:, 0].ptp(), all_arr[:, 1].ptp()) or 1
    scale0   = min(SCREEN_W, SCREEN_H) * 0.8 / span

    # ── path .txt 자동 탐색 ──
    # Path file is rewritten by scenario_runner whenever a new route is selected.
    # Reload it automatically so the visualization follows the current spawn/route.
    path_pts = []
    path_file = os.path.join(current_path, 'path_scenario_runner_current.txt')
    path_mtime = None

    def reload_path_if_changed(force=False):
        nonlocal path_pts, path_mtime
        if not os.path.exists(path_file):
            return

        mtime = os.path.getmtime(path_file)
        if not force and path_mtime == mtime:
            return

        path_mtime = mtime
        path_pts = load_path_txt(path_file)
        print(f'경로 파일 로드: {os.path.basename(path_file)}  ({len(path_pts)} waypoints)')

    reload_path_if_changed(force=True)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption('MGeo Visualization')
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont('consolas', 14)

    offset     = [cx, cy]
    scale      = scale0
    drag_start = None
    drag_offset= None

    running = True
    while running and not rospy.is_shutdown():
        reload_path_if_changed()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                drag_start  = event.pos
                drag_offset = offset[:]

            elif event.type == pygame.MOUSEMOTION and drag_start:
                dx = (event.pos[0] - drag_start[0]) / scale
                dy = (event.pos[1] - drag_start[1]) / scale
                offset[0] = drag_offset[0] - dx
                offset[1] = drag_offset[1] + dy

            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                drag_start = None

            elif event.type == pygame.MOUSEWHEEL:
                scale *= 1.1 ** event.y

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                offset = [cx, cy]
                scale  = scale0

        screen.fill(COLOR_BG)

        # 링크
        for pts in link_polylines:
            if len(pts) < 2:
                continue
            scr = world_to_screen(pts, offset, scale)
            pygame.draw.lines(screen, COLOR_LINK, False, scr.tolist(), 1)

        # 노드
        scr_nodes = world_to_screen(node_pts, offset, scale)
        for p in scr_nodes:
            pygame.draw.circle(screen, COLOR_NODE, p, 3)

        # 경로
        if len(path_pts) >= 2:
            scr_path = world_to_screen(path_pts, offset, scale)
            pygame.draw.lines(screen, COLOR_PATH, False, scr_path.tolist(), 3)
            pygame.draw.circle(screen, COLOR_START, scr_path[0],  8)
            pygame.draw.circle(screen, COLOR_END,   scr_path[-1], 8)

        # 차량 위치
        with vehicle_lock:
            vx, vy, vyaw = vehicle_pos['x'], vehicle_pos['y'], vehicle_pos['yaw']

        if vx is not None:
            scr_v = world_to_screen([(vx, vy)], offset, scale)[0]
            draw_vehicle(screen, scr_v[0], scr_v[1], vyaw, scale)

        # HUD
        with vehicle_lock:
            pos_txt = f'차량 위치: ({vx:.1f}, {vy:.1f})' if vx is not None else '차량 위치: 수신 대기 중...'
        lines_hud = [
            'R: 초기화   스크롤: 줌   드래그: 이동',
            f'노드 {len(nodes)}   링크 {len(links)}',
            pos_txt,
        ]
        if path_pts:
            lines_hud.append(f'경로 waypoints: {len(path_pts)}  ● 시작  ● 종료')
        for i, txt in enumerate(lines_hud):
            screen.blit(font.render(txt, True, COLOR_TEXT), (10, 10 + i * 18))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == '__main__':
    main()
