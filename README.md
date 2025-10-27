# Közlekedési Objektumkövetés és Detektálás

## Projekt Áttekintés
Ez a projekt egy közlekedési objektumkövető és -detektáló rendszer, amely képes mind hagyományos számítógépes látás alapú, mind mesterséges intelligencia alapú objektumkövetésre. A rendszer YOLO (You Only Look Once) és egyéni követő algoritmusokat használ.

## Funkciók
- Kétirányú forgalomszámlálás
- AI-alapú objektumdetektálás (YOLO)
- Hagyományos objektumkövetés (MOG2 háttérkivonás)
- Valós idejű követés és megjelenítés
- YOLO és COCO formátumú adathalmazok kezelése

## Telepítési Útmutató

### Követelmények
```bash
pip install -r requirements.txt
```

Főbb függőségek:
- Python 3.8+
- PyTorch
- OpenCV
- Ultralytics YOLO
- NumPy
- Pandas

### Adathalmaz Előkészítése
Az adathalmaz készen áll yolo11 modelek tanításához.

## Használat

### Objektumkövetés Futtatása
```bash
python tracking.py --ai --input video.mp4 --display True
```

Paraméterek:
- `--ai`: AI-alapú követés használata (alapértelmezett: False)
- `--input`: Bemeneti videófájl (alapértelmezett: "video.mp4")
- `--display`: Megjelenítés engedélyezése (alapértelmezett: True)
- `--output`: Kimeneti videófájl (opcionális)

### YOLO Modell Tanítása
```bash
python train_yolo.py
```

### Adathalmaz Konvertálása
YOLO formátumból COCO formátumba való konvertáláshoz:
```bash
python converttoyolo.py
```

## Projekt Struktúra
```
.
├── data/
│   └── trafficcam/           # YOLO formátumú adathalmaz
├── weights/                  # Tanított modellek
├── tracking.py              # Fő követő alkalmazás
├── train_yolo.py            # YOLO tanító script
├── converttoyolo.ipynb         # Formátum konvertáló
└── DistanceTracking.py      # Követő algoritmus
```

## Modellek
A `weights` mappában található előre tanított modellek:
- `trafic_5.pt`: 5 epchon át rátanított YOLO11N model

## Hibakezelés

### Gyakori Problémák és Megoldások
1. CUDA memória hiba esetén:
   - Csökkentse a batch méretet
   - Használjon kisebb képméretet
   - Állítsa át CPU módra

2. DLL betöltési hiba esetén:
   - Telepítse a Visual C++ Redistributable-t
   - Ellenőrizze a CUDA verziókat
   - Használjon CPU-only PyTorch-ot

## Fejlesztői Megjegyzések
- A modell teljesítménye függ a GPU kapacitástól
- Windows környezetben ajánlott a workers=0 használata
- Nagyobb adathalmazok esetén javasolt a batch méret optimalizálása