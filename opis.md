
# AuraCloud

## 1. Executive Summary (O co chodzi?)
Projekt to zaawansowana aplikacja typu **BCI (Brain-Computer Interface)**, której celem jest mostowanie przepaści między biologicznymi sygnałami mózgu a cyfrową interpretacją emocji. System w czasie rzeczywistym analizuje sygnał EEG, mapuje go na trójwymiarową przestrzeń emocjonalną, a następnie zamyka pętlę sprzężenia zwrotnego poprzez precyzyjną stymulację wizualną.

To nie jest tylko "czytnik nastroju" – to narzędzie do **cyfrowej kwantyfikacji i modulacji stanów afektywnych**.

## 2. Architektura Systemu i Emocje (Core Mechanics)

### Model Teoretyczny
Aplikacja opiera się na **Trójczynnikowej Teorii Emocji (PAD Model)**. Każdy stan emocjonalny użytkownika jest wektorem w przestrzeni 3D zdefiniowanym przez:
* **Valence (Walencja):** Oś przyjemności (od negatywnej do pozytywnej).
* **Arousal (Pobudzenie):** Oś energii (od uśpienia do ekscytacji).
* **Dominance (Dominacja):** Oś kontroli (od uległości do poczucia władzy).

### Wizualizacja: "The Neuro-Cloud"
Zamiast nudnych słupków, aplikacja generuje **interaktywną chmurę punktów w przestrzeni 3D**.
* **Catchy Feature:** Wyobraź sobie pulsującą, "żywą" mgławicę, która zmienia kolor i dynamikę ruchu w zależności od Twojego mózgu.
    * *Gniew/Stres:* Chmura staje się czerwona, szybka i ostra (High Arousal, Low Valence).
    * *Relaks:* Chmura zmienia się w błękitną, wolno falującą taflę (Low Arousal, High Valence).
To daje użytkownikowi natychmiastowy, intuicyjny **Biofeedback**.

## 3. Moduł Stymulacji (The Stimuli Engine)

Aby nie tylko "czytać", ale i "kalibrować" emocje, stworzyliśmy autorski podzbiór danych oparty o zwalidowany naukowo dataset **OASIS**.

### Nasz "Złoty Zbiór" (Calibration Dataset)
Z tysięcy obrazów wyekstrahowaliśmy **100 najsilniejszych bodźców wizualnych** (tzw. kotwic), które gwarantują wywołanie konkretnej reakcji.
* **Zakres danych:**
    * Valence: $1.11$ (skrajny negatyw) $\rightarrow$ $6.49$ (skrajna euforia)
    * Arousal: $1.69$ (głęboki spokój) $\rightarrow$ $5.47$ (wysoki szok/ekscytacja)

### Klasyfikacja Emocji (4 Ćwiartki + Centrum)
System operuje na mapowaniu 2D (Valence/Arousal) z założeniem neutralnej Dominacji dla bodźców statycznych:

1.  **JOY 🟢:** (High Valence, High Arousal) – Radość, Ekscytacja.
2.  **FEAR 🔴:** (Low Valence, High Arousal) – Stres, Strach, Wstręt.
3.  **RELAX 🔵:** (High Valence, Low Arousal) – Spokój, Ukojenie.
4.  **SADNESS ⚫:** (Low Valence, Low Arousal) – Smutek, Melancholia, Znużenie.
5.  **NEUTRAL ⚪:** (Mid Valence, Mid Arousal) – Punkt odniesienia (Baseline).



## 4. Funkcjonalności (Roadmap)

### Faza 1: Analiza i Pasywna Stymulacja (Obecnie)
* **Input:** Sygnał EEG z czepka.
* **Process:** Dekodowanie fal mózgowych $\rightarrow$ Mapowanie na model PAD.
* **Output:** Wizualizacja 3D w czasie rzeczywistym.
* **Kalibracja:** Wyświetlanie sekwencji zdjęć ze "Złotego Zbioru" w celu zbadania reaktywności użytkownika na bodźce.

### Faza 2: "Mood Inducer" (Wkrótce)
* Moduł odwrotny: **User $\rightarrow$ App**.
* Użytkownik za pomocą dwóch sliderów określa stan docelowy (np. "Chcę się poczuć zrelaksowany, ale pełen energii").
* Algorytm dobiera z bazy obraz (lub sekwencję), który posiada parametry matematycznie najbliższe żądanym wartościom suwaków.

## 5. Zastosowania i Wartość Biznesowa

Projekt wykracza poza ramy akademickiego eksperymentu, oferując realną wartość w:

* **Psychoterapia i Leczenie Fobii:**
    * Wykorzystanie w terapii ekspozycyjnej (desensytyzacja). Monitorowanie w czasie rzeczywistym, czy pacjent faktycznie się uspokaja, czy tylko "mówi", że jest spokojny.
* **Neuromarketing:**
    * Obiektywna ocena kampanii reklamowych. Badanie siły reakcji emocjonalnej (Arousal) i jej kierunku (Valence) z pominięciem błędów deklaratywnych (kłamania w ankietach).
* **Trening Mentalny (High-Performance):**
    * Dla sportowców: nauka wchodzenia w stan "Flow" poprzez wizualny biofeedback.
