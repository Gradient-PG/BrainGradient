from emotion_image_selector import get_closest_theme, load_oasis_dataset

# jednorazowo wczytujesz dataset, żeby nie ładować go za każdym wywołaniem:
df = load_oasis_dataset("final_eeg_dataset.csv")

# użytkownik ustawia suwaki na (0.8, 0.3) – dużo pozytywnej walencji, umiarkowane pobudzenie
theme_name = get_closest_theme(1, 1, df=df)
print(theme_name)  # np. "Lake 12.jpg"
