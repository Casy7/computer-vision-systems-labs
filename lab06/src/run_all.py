"""
Лабораторна робота № 6: «Оптичний потік і відстеження об'єктів».
Головний скрипт послідовного виконання усіх етапів роботи (Частини A–E, Кроки 1–19).
"""

import time
from part_a_b import run_part_a_b
from part_c import run_part_c
from part_d import run_part_d
from part_e import run_part_e


def main():
    t0 = time.perf_counter()
    run_part_a_b()
    run_part_c()
    run_part_d()
    run_part_e()
    dt = time.perf_counter() - t0
    print("=" * 65)
    print(f"Усі етапи Лабораторної роботи № 6 успішно виконано за {dt:.1f} с.")
    print("=" * 65)


if __name__ == "__main__":
    main()
