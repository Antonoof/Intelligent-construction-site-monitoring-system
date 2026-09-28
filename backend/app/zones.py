"""Привязка техники к зонам кадра.

Машина относится к зоне, в которую попадает её точка опоры — середина нижней стороны рамки:
так высокая стрела крана или экскаватора не «переносит» машину в соседнюю зону. Полигоны зон
хранятся в долях кадра (0..1), поэтому не зависят от разрешения снимка.
"""
from __future__ import annotations


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def ground_point(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2 / width, min(y2, height - 1) / height)


def assign_zone(box, width: int, height: int, polygons: list[tuple[str, list]], default: str | None) -> str | None:
    """Зона по точке опоры; если точка вне всех полигонов — зона камеры по умолчанию."""
    gx, gy = ground_point(box, width, height)
    for key, poly in polygons:
        if point_in_polygon(gx, gy, poly):
            return key
    # точка опоры чуть ниже края полигона (обрез кадра) — проверяем центр рамки
    cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
    for key, poly in polygons:
        if point_in_polygon(cx, cy, poly):
            return key
    return default
