from emotion_image_selector import (
    load_oasis_dataset,
    get_closest_theme,
    generate_path,
)

df = load_oasis_dataset()

tar_a, tar_v, tar_d = 0.2, 0.8, 0.5
cur_a, cur_v, cur_d = 0.7, 0.3, 0.4
step = 0.1

recent = []  # tu trzymasz historię Theme’ów

while True:
    next_a, next_v, reached = generate_path(
        tar_a, tar_v, tar_d,
        cur_a, cur_v, cur_d,
        step,
    )

    theme = get_closest_theme(
        v_norm=next_v,
        a_norm=next_a,
        df=df,
        recent_themes=recent,
        avoid_repeats=True,  # ustaw False, żeby wyłączyć ograniczenie
        repeat_window=3,
    )

    print("Wyświetlam:", theme)
    recent.append(theme)

    # aktualizacja z EEG – tutaj na razie tylko placeholder:
    cur_a, cur_v = next_a, next_v

    if reached:
        break
