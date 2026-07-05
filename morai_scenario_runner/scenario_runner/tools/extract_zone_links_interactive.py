import json
import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


def load_link_set(mgeo_root):
    path = os.path.join(mgeo_root, "link_set.json")
    with open(path, "r") as f:
        data = json.load(f)
    return {item["idx"]: item for item in data}


def link_centroid(points):
    arr = np.array(points)
    return arr[:, 0].mean(), arr[:, 1].mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgeo-root", required=True)
    parser.add_argument("--output", default="scenario_runner/config/highway_links.yaml")
    parser.add_argument("--zone-name", default="highway")
    args = parser.parse_args()

    links = load_link_set(args.mgeo_root)

    fig, ax = plt.subplots(figsize=(12, 10))
    clicked = []
    point_artists = []
    polygon_line = None
    pan_state = {"pressed": False, "x": 0, "y": 0, "xlim": None, "ylim": None}

    for link_id, link in links.items():
        pts = np.array(link["points"])
        ax.plot(pts[:, 0], pts[:, 1], linewidth=0.7)
        cx, cy = link_centroid(link["points"])
        ax.text(cx, cy, link_id, fontsize=5)

    ax.set_aspect("equal")
    ax.set_title(
        "Left click: add point | Backspace: undo | Enter: save | Mouse wheel: zoom | Right drag: pan"
    )

    def redraw_polygon():
        nonlocal polygon_line
        if polygon_line:
            polygon_line.remove()
            polygon_line = None

        if len(clicked) >= 2:
            xs = [p[0] for p in clicked]
            ys = [p[1] for p in clicked]
            if len(clicked) >= 3:
                xs.append(clicked[0][0])
                ys.append(clicked[0][1])
            polygon_line, = ax.plot(xs, ys, "r-", linewidth=2)

        fig.canvas.draw_idle()

    def on_scroll(event):
        if event.inaxes != ax:
            return

        base = 1.2
        scale = 1 / base if event.button == "up" else base

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_w = (xlim[1] - xlim[0]) * scale
        new_h = (ylim[1] - ylim[0]) * scale

        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - new_w * (1 - relx), xdata + new_w * relx])
        ax.set_ylim([ydata - new_h * (1 - rely), ydata + new_h * rely])
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes != ax:
            return

        if event.button == 1:
            clicked.append((event.xdata, event.ydata))
            point_artists.append(ax.plot(event.xdata, event.ydata, "ro", markersize=5)[0])
            print(f"point {len(clicked)}: ({event.xdata:.2f}, {event.ydata:.2f})")
            redraw_polygon()

        elif event.button == 3:
            pan_state["pressed"] = True
            pan_state["x"] = event.xdata
            pan_state["y"] = event.ydata
            pan_state["xlim"] = ax.get_xlim()
            pan_state["ylim"] = ax.get_ylim()

    def on_release(event):
        if event.button == 3:
            pan_state["pressed"] = False

    def on_motion(event):
        if not pan_state["pressed"] or event.inaxes != ax:
            return

        dx = event.xdata - pan_state["x"]
        dy = event.ydata - pan_state["y"]

        x0, x1 = pan_state["xlim"]
        y0, y1 = pan_state["ylim"]
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "backspace":
            if clicked:
                clicked.pop()
                artist = point_artists.pop()
                artist.remove()
                redraw_polygon()

        elif event.key == "enter":
            if len(clicked) < 3:
                print("Need at least 3 points.")
                return

            polygon = Path(clicked)
            selected_links = []

            for link_id, link in links.items():
                cx, cy = link_centroid(link["points"])
                if polygon.contains_point((cx, cy)):
                    selected_links.append(link_id)

            selected_links = sorted(selected_links)

            result = {
                args.zone_name: {
                    "polygon": [[float(x), float(y)] for x, y in clicked],
                    "route_links": selected_links,
                    "exclude_links": [],
                    "event_links": {
                        "sudden_brake": [],
                        "obstacle": [],
                        "pedestrian": [],
                    },
                }
            }

            with open(args.output, "w") as f:
                yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)

            print(f"Saved: {args.output}")
            print(f"Selected links: {len(selected_links)}")
            for link_id in selected_links:
                print(link_id)

            plt.close(fig)

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()


if __name__ == "__main__":
    main()
