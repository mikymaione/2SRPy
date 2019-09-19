# Implementazione in Python di "Spatial Subspace Rotation"

In "A Novel Algorithm for Remote Photoplethysmography - Spatial Subspace Rotation" a cura di W. Wang, S. Stuijk e G. de Haan, gli autori propongono un algoritmo (2SR) per la fotopletismografia remota (rPPG).

L'algorimo 2SR richiede il riconoscimento degli skin-pixels, e per lo scopo è stato utilizzato l'algoritmo (SkinColorFilter) proposto in "Adaptive skin segmentation via feature-based face detection" a cura di M.J. Taylor e T. Morris.

L'algoritmo SkinColorFilter riceve in ingresso un volto che deve essere estratto da un filmato. L'estrazione del volto è stata ottenuta mediante il classificatore a cascata Haar proposto da P. Viola e M. Jones in "Rapid Object Detection using a Boosted Cascade of Simple Features".

Ambedue gli algoritmi sono stati implementati in classi Python: SSR.py, SkinColorFilter.py. Mentre per l'estrazione del volto è stata usata la libreria OpenCV.

## IDE
PyCharm 2018 (http://www.jetbrains.com/pycharm)

## License
Copyright 2019 (c) Maione Michele. All rights reserved.

Licensed under the [MIT](LICENSE) License.
