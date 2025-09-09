# Active Learning für Bildklassifikation - Bachelorarbeit

## Überblick

Dieses Repository enthält die Implementierung und experimentelle Evaluierung verschiedener Active Learning Strategien für die Bildklassifikation. Die Arbeit untersucht, wie Machine Learning Modelle durch intelligente Auswahl von Trainingsdaten effizienter trainiert werden können, wodurch der Annotationsaufwand erheblich reduziert wird.

## Erstellungshinweis

**Dieses Repository wurde mit Unterstützung von Claude Opus 4 (Anthropic) entwickelt.** Claude Opus 4 hat bei der Konzeption, Implementierung und Dokumentation des Codes sowie bei der Strukturierung der Active Learning Strategien assistiert.

## Inhaltsverzeichnis

- [Projektstruktur](#projektstruktur)
- [Untersuchte Datensätze](#untersuchte-datensätze)
- [Implementierte Algorithmen](#implementierte-algorithmen)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Ergebnisse](#ergebnisse)
- [Technische Details](#technische-details)

## Projektstruktur

```
bachelorarbeit/
│
├── Jupyter_Notebooks_MNIST/          # MNIST Experimente
│   ├── CNN_MNIST.ipynb               # Convolutional Neural Network
│   ├── LR_MNIST.ipynb                # Logistic Regression
│   ├── NB_MNIST.ipynb                # Naive Bayes
│   ├── RF_MNIST.ipynb                # Random Forest
│   ├── SVM_MNIST.ipynb               # Support Vector Machine
│   └── Plots_MNIST.ipynb             # Visualisierungen
│
├── Jupyter_Notebooks_Fashion_MNIST/   # Fashion-MNIST Experimente
│   ├── CNN_Fashion_MNIST.ipynb       # Convolutional Neural Network
│   ├── LR_Fashion_MNIST.ipynb        # Logistic Regression
│   ├── NB_Fashion_MNIST.ipynb        # Naive Bayes
│   ├── RF_Fashion_MNIST.ipynb        # Random Forest
│   └── SVM_Fashion_MNIST.ipynb       # Support Vector Machine
│
├── Dachmaterialien_F1_Score.ipynb    # Dachmaterial-Klassifikation (Hauptexperiment)
│
├── Results_MNIST/                     # MNIST Ergebnisse
│   ├── gpu_*_active_learning_results.csv
│   ├── gpu_*_active_learning_summary.xlsx
│   ├── gpu_*_label_savings.csv
│   └── gpu_*_statistical_analysis.csv
│
├── Results_Fashion_MNIST/             # Fashion-MNIST Ergebnisse
│   ├── fashion_*_active_learning_results.csv
│   ├── fashion_*_active_learning_summary.xlsx
│   ├── fashion_*_label_savings.csv
│   └── fashion_*_statistical_analysis.csv
│
├── Dachmaterialien_Results/           # Dachmaterial Ergebnisse
│   ├── dachmaterial_f1_active_learning_ergebnisse.csv
│   ├── dachmaterial_f1_active_learning_zusammenfassung.xlsx
│   ├── dachmaterial_f1_label_einsparungen.csv
│   └── dachmaterial_f1_statistische_analyse.csv
│
├── Plots_MNIST/                       # MNIST Visualisierungen
├── Plots_Fashion_MNIST/               # Fashion-MNIST Visualisierungen
├── Dachmaterialien_Plots/             # Dachmaterial Visualisierungen
│
├── umrisse_with_all_data_and_shape_and_patch_and_normal.csv  # Dachmaterial Rohdaten
└── requirements.txt                   # Python Abhängigkeiten
```

## Untersuchte Datensätze

### 1. MNIST
- **Beschreibung**: Handgeschriebene Ziffern (0-9)
- **Anzahl Klassen**: 10
- **Bildgröße**: 28x28 Pixel (Graustufen)
- **Trainingsdaten**: 60.000 Bilder
- **Testdaten**: 10.000 Bilder

### 2. Fashion-MNIST
- **Beschreibung**: Kleidungsstücke und Accessoires
- **Anzahl Klassen**: 10 (T-Shirt, Hose, Pullover, Kleid, Mantel, Sandale, Hemd, Sneaker, Tasche, Stiefelette)
- **Bildgröße**: 28x28 Pixel (Graustufen)
- **Trainingsdaten**: 60.000 Bilder
- **Testdaten**: 10.000 Bilder

### 3. Dachmaterialien (Custom Dataset)
- **Beschreibung**: Satellitenbilder von Dächern mit verschiedenen Materialien
- **Anzahl Klassen**: Variabel (z.B. Beton, Bitumen, Ziegel, etc.)
- **Features**: Geometrische und spektrale Eigenschaften
- **Besonderheit**: Stark unbalancierter Datensatz
- **Hauptmetrik**: F1-Score (Macro) aufgrund der Klassenungleichgewichte

## Implementierte Algorithmen

### Machine Learning Modelle
1. **Convolutional Neural Networks (CNN)**: Deep Learning Ansatz für Bildklassifikation
2. **Logistic Regression (LR)**: Baseline-Klassifikator
3. **Naive Bayes (NB)**: Probabilistisches Modell
4. **Random Forest (RF)**: Ensemble-Methode
5. **Support Vector Machine (SVM)**: Kernel-basierte Klassifikation

### Active Learning Strategien
- **Random Sampling**: Zufällige Auswahl (Baseline)
- **Uncertainty Sampling**: Auswahl der unsichersten Samples
- **Entropy-based Sampling**: Maximierung der Entropie
- **Margin Sampling**: Minimaler Abstand zur Entscheidungsgrenze
- **Query-by-Committee**: Ensemble-basierte Unsicherheit

## Installation

### Voraussetzungen
- Python 3.8 oder höher
- CUDA-fähige GPU (empfohlen für CNN-Training)
- Mindestens 16 GB RAM

### Abhängigkeiten installieren

```bash
# Repository klonen
git clone https://github.com/[username]/bachelorarbeit.git
cd bachelorarbeit

# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder
venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### Wichtige Bibliotheken
- **PyTorch**: Deep Learning Framework
- **scikit-learn**: Machine Learning Algorithmen
- **NumPy/Pandas**: Datenverarbeitung
- **Matplotlib/Seaborn**: Visualisierung
- **CUDA/cuDNN**: GPU-Beschleunigung
- **Jupyter**: Interaktive Notebooks

## Verwendung

### 1. MNIST Experimente ausführen

```python
# Öffne Jupyter Notebook
jupyter notebook

# Navigiere zu Jupyter_Notebooks_MNIST/
# Öffne und führe die gewünschten Notebooks aus:
# - CNN_MNIST.ipynb für CNN-Experimente
# - LR_MNIST.ipynb für Logistic Regression
# etc.
```

### 2. Fashion-MNIST Experimente

```python
# Navigiere zu Jupyter_Notebooks_Fashion_MNIST/
# Führe die entsprechenden Notebooks aus
```

### 3. Dachmaterial-Klassifikation

```python
# Öffne Dachmaterialien_F1_Score.ipynb
# Dieses Notebook enthält die komplette Pipeline:
# - Datenvorbereitung
# - Active Learning Experimente
# - Statistische Analyse
# - Visualisierung
```

### Konfigurationsparameter

Die wichtigsten Parameter können in den Notebooks angepasst werden:

```python
# Beispiel-Konfiguration
CONFIG = {
    'n_runs': 10,              # Anzahl der Durchläufe
    'n_queries': 20,           # Anzahl der Active Learning Iterationen
    'batch_size': 100,         # Samples pro Iteration
    'initial_samples': 100,    # Initiale Trainingsdaten
    'test_size': 0.3,         # Anteil der Testdaten
    'random_state': 42        # Für Reproduzierbarkeit
}
```

## Ergebnisse

### Ausgabeformate

#### CSV-Dateien
- **active_learning_results.csv**: Detaillierte Ergebnisse aller Durchläufe
- **label_savings.csv**: Analyse der Annotationseinsparungen
- **statistical_analysis.csv**: Statistische Tests und Signifikanz

#### Excel-Dateien
- **active_learning_summary.xlsx**: Zusammenfassung mit mehreren Tabellenblättern

#### Visualisierungen
- **Performance-Plots**: Lernkurven über Active Learning Iterationen
- **Statistische Analyse**: Box-Plots und Signifikanztests
- **Label-Einsparungen**: Vergleich zum Random Sampling
- **Vergleichsanalysen**: Gegenüberstellung aller Modelle

### Metriken

#### Für balancierte Datensätze (MNIST, Fashion-MNIST)
- **Accuracy**: Hauptmetrik
- **Precision/Recall**: Pro Klasse
- **Confusion Matrix**: Fehleranalyse

#### Für unbalancierte Datensätze (Dachmaterialien)
- **F1-Score (Macro)**: Hauptmetrik
- **Precision/Recall (Macro)**: Klassenübergreifend
- **Class-wise F1-Scores**: Pro Material

### Statistische Auswertung
- **Wilcoxon Signed-Rank Test**: Signifikanz der Verbesserungen
- **Cliff's Delta**: Effektstärke
- **Konfidenzintervalle**: 95% CI für alle Metriken
- **p-Werte**: Statistische Signifikanz (p < 0.05)

## Technische Details

### GPU-Beschleunigung
Das Projekt nutzt CUDA für beschleunigtes Training:
- Automatische GPU-Erkennung
- Mixed Precision Training (wo unterstützt)
- Batch-Processing für große Datensätze

### Reproduzierbarkeit
Alle Experimente sind reproduzierbar durch:
- Feste Random Seeds
- Deterministisches Training
- Versionskontrolle der Abhängigkeiten

### Logging
Detailliertes Logging in `logs/`:
- Training-Fortschritt
- Fehlerbehandlung
- Performance-Metriken

## Wichtige Erkenntnisse

1. **Active Learning Effizienz**: Reduzierung des Annotationsaufwands um bis zu 70% bei gleicher Performance
2. **Modellabhängigkeit**: CNN profitiert am meisten von Active Learning
3. **Datensatzcharakteristika**: Unbalancierte Datensätze erfordern angepasste Metriken (F1-Score)
4. **Query-Strategien**: Uncertainty Sampling zeigt konsistent gute Ergebnisse

## Erweiterungsmöglichkeiten

- Integration weiterer Query-Strategien (z.B. BALD, Core-Set)
- Transfer Learning Ansätze
- Semi-Supervised Learning Kombinationen
- Online Active Learning
- Multi-Class Active Learning Optimierungen

## Kontakt und Support

Bei Fragen oder Problemen:
- Issues im GitHub Repository erstellen
- Dokumentation in den Jupyter Notebooks konsultieren
- Code-Kommentare für Implementierungsdetails

## Lizenz

Dieses Projekt wurde im Rahmen einer Bachelorarbeit erstellt. Die Verwendung für akademische Zwecke ist gestattet mit entsprechender Zitierung.

---

*Entwickelt im Rahmen der Bachelorarbeit zum Thema "Effiziente Generierung von Trainingsdaten in der Bildklassifikation"*
