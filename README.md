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

## Paraméterezhetőség és Finomhangolás traditional_car.py

A rendszer a CustomTkinter (CTk) vezérlőpulton keresztül több paramétert is lehetővé tesz, amelyekkel a videófeldolgozás hagyományos, **MOG2 alapú** módja finomhangolható. Ezek a paraméterek kritikusak a pontos detektálás és követés szempontjából, és eltérő fényviszonyokhoz, videófelbontásokhoz vagy objektumtípusokhoz állíthatók be.

### 1\. Általános Számlálási Paraméterek

| **Paraméter** | **Mire jó?** | **Kisebb érték esetén** | **Nagyobb érték esetén** |
| --- | --- | --- | --- |
| **LINE_Y** | Meghatározza a számlálóvonal függőleges **Y koordinátáját** (pixelben) a videóban. Ez a vonal detektálja az áthaladást. | A vonal feljebb kerül, így **hamarabb** számlálja a magasabb objektumokat, vagy a kép felső részén lévő áthaladást. | A vonal lejjebb kerül, így a kamerához közelebb/később számlál, vagy kihagyja a túl kicsi/magas objektumokat. |
| --- | --- | --- | --- |
| **MIN_AREA** | A minimális **pixelfelület** (px²) méret, amit egy kontúrnak el kell érnie ahhoz, hogy detektálásnak minősüljön a háttérkivonás után. | Több kisebb **zaj** vagy **árnyék** is objektumként detektálódik, növelve a hamis pozitívok számát. | Csak a nagy objektumok (pl. teherautók) detektálódnak, a kisebbek (pl. biciklik, távoli autók) kimaradnak. |
| --- | --- | --- | --- |

### 2\. MOG2 Háttérkivonó Paraméterei

A MOG2 (Mixture of Gaussians) algoritmus a háttér és az előtér mozgó objektumai közötti különbséget érzékeli.

| **Paraméter** | **Mire jó?** | **Kisebb érték esetén** | **Nagyobb érték esetén** |
| --- | --- | --- | --- |
| **MOG2_HISTORY** (Hist) | Az az időtartam (képkockák száma), ameddig a háttérkivonó az előző képkockákat figyelembe veszi a **háttér modell** építéséhez. | A háttér **gyorsabban frissül/adaptálódik**. Gyorsan mozgó statikus objektumok (pl. egy álló autó) gyorsabban háttérré válnak. | A háttér **lassabban adaptálódik**. Tökéletes, ha a háttér stabil, de mozgó árnyékok esetén detektálási hibát okozhat. |
| --- | --- | --- | --- |
| **MOG2_THRESHOLD** (Thresh) | A pixelszín különbségének **érzékenységi küszöbe** a háttérmodellhez képest (varThreshold). | Nagyobb érzékenység: a detektor a háttértől való apró eltérésekre (zaj, fényváltozás, **árnyék**) is reagál, ami zajosabb detektálást eredményez. | Kisebb érzékenység: csak a nagyon **markáns mozgást** veszi figyelembe. Csökkenti a zajt, de az objektumok szélei kieshetnek. |
| --- | --- | --- | --- |

### 3\. Morfológiai Paraméterek

A Morfológiai Műveletek a háttérkivonás zaját és a detektált kontúrok hibáit tisztítják.

| **Paraméter** | **Mire jó?** | **Kisebb érték esetén** | **Nagyobb érték esetén** |
| --- | --- | --- | --- |
| **MORPH_OPEN** (Open It.) | **Nyitás (Opening) iterációi.** Eltávolítja a maszkból a zajt (apró pozitívumokat/pöttyöket), majd helyreállítja a kontúrt. | Kevésbé hatékony a maszkban lévő **zaj eltávolításában**. | Túl sok Nyitás esetén a valódi objektumok maszkjai is **összezsugorodnak** vagy eltűnnek. |
| --- | --- | --- | --- |
| **MORPH_CLOSE** (Close It.) | **Zárás (Closing) iterációi.** Bezárja a maszkban lévő apró lyukakat, és **összeköti a közeli mozgó részeket**, így egy mozgó objektum egy kontúrként jelenik meg. | Nem köti össze megfelelően az egy objektumhoz tartozó, de széteső mozgó részeket. | Túl sok Zárás esetén az egymáshoz **közeli, különálló objektumok összekapcsolódnak** egyetlen kontúrrá (hibás számlálás). |
| --- | --- | --- | --- |

### 4\. Objektumkövető Paraméterek

| **Paraméter** | **Mire jó?** | **Kisebb érték esetén** | **Nagyobb érték esetén** |
| --- | --- | --- | --- |
| **TRACKER_DISTANCE** (Tracker Dist) | Az **EuclideanDistTracker** maximális távolsága (pixelben) két képkocka között. Ha egy detektált centrumpont ezen a távolságon belül esik egy meglévő ID-hez képest, akkor azonos objektumként kezeli. | A követő rendszer **túl szigorú** lesz; a gyorsan mozgó vagy pillanatra eltűnő objektumok új ID-t kapnak (azaz szétesik a követésük). | A követő rendszer **túl megengedő** lesz; az egymáshoz közel elhaladó, különböző objektumokat (pl. szorosan követő autók) tévesen azonos objektumnak kezeli, **hibás számlálást** okozva. |
| --- | --- | --- | --- |