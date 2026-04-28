"""Convert ISIC dataset images from RGB to the HED (Hematoxylin–Eosin–DAB) color space.

Espelha a estrutura de `rgb_to_lab.py`, percorrendo a árvore de pastas do
dataset original, copiando todos os arquivos auxiliares (labels .txt,
data.yaml, README, etc.) e convertendo apenas as imagens dentro de pastas
chamadas `images`.

Implementação:
    1. Lê a imagem com OpenCV (BGR uint8) e converte para RGB.
    2. Aplica `skimage.color.rgb2hed` (deconvolução de Ruifrok-Johnston),
       que retorna um array float com 3 canais (H, E, D) em escala de
       densidade óptica.
    3. Reescala cada canal independentemente para uint8 [0, 255] usando
       `skimage.exposure.rescale_intensity` (min-max por canal por imagem),
       gerando uma imagem visualizável "RGB-like" onde:
         * canal R (cv2 BGR=2) ← H (Hematoxylin)
         * canal G (cv2 BGR=1) ← E (Eosin)
         * canal B (cv2 BGR=0) ← D (DAB)
    4. Salva via `cv2.imwrite` (espera BGR uint8).

Observações:
    - ISIC é dermatoscopia, não slide H&E histológico — a hipótese original
      do HED (separar coloração hematoxilina/eosina/DAB) não se aplica
      diretamente. Os valores absolutos por canal não terão significado
      físico aqui, mas a transformação ainda é determinística e
      descorrelaciona os canais, podendo servir como representação
      alternativa para o fine-tuning, similar ao que já fizemos com LAB.
    - A normalização é por imagem (min-max) — escolha que mantém contraste
      visual; uma alternativa seria fixar a faixa global em [0, 1] vinda
      direto do `rgb2hed`, mas isso costuma resultar em imagens muito
      escuras/saturadas para inputs não-histológicos.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
from skimage.color import rgb2hed
from skimage.exposure import rescale_intensity


def convert_rgb_to_hed_uint8(img_bgr: np.ndarray) -> np.ndarray:
    """Convert a BGR uint8 image to a HED-encoded BGR uint8 image.

    Each HED channel is independently min-max rescaled to [0, 255] within
    the same image. The output is encoded as BGR so that `cv2.imwrite`
    saves a sensible 3-channel image (B=DAB, G=Eosin, R=Hematoxylin).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hed = rgb2hed(img_rgb)  # float, shape (H, W, 3): channels = H, E, D

    h_u8 = rescale_intensity(hed[:, :, 0], out_range=(0, 255)).astype(np.uint8)
    e_u8 = rescale_intensity(hed[:, :, 1], out_range=(0, 255)).astype(np.uint8)
    d_u8 = rescale_intensity(hed[:, :, 2], out_range=(0, 255)).astype(np.uint8)

    # cv2 espera BGR -> empilha (D, E, H) para virar (B=D, G=E, R=H)
    return np.dstack([d_u8, e_u8, h_u8])


def main() -> None:
    # Caminhos da base original e da nova base convertida
    input_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26")
    output_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26_hed")

    # Extensões de imagem que queremos converter
    image_extensions = {".jpg", ".jpeg", ".png"}

    print(f"Iniciando conversão para HED...\nOrigem: {input_base}\nDestino: {output_base}\n")

    # Percorrer toda a estrutura de pastas e ficheiros
    for root, dirs, files in os.walk(input_base):
        # Calcular o caminho relativo para replicar a mesma estrutura no destino
        rel_path = Path(root).relative_to(input_base)
        dest_dir = output_base / rel_path

        # Criar a pasta de destino (ex: train/images, train/labels, etc.)
        dest_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            src_file = Path(root) / file
            dest_file = dest_dir / file

            # Se for uma imagem E estiver dentro de uma pasta chamada "images"
            if src_file.suffix.lower() in image_extensions and "images" in src_file.parts:
                # 1. Lê a imagem (o OpenCV carrega em BGR)
                img_bgr = cv2.imread(str(src_file))

                if img_bgr is not None:
                    # 2. Converte de BGR (passa por RGB) para HED uint8 (BGR-encoded)
                    img_hed = convert_rgb_to_hed_uint8(img_bgr)

                    # 3. Guarda a imagem convertida na nova pasta
                    cv2.imwrite(str(dest_file), img_hed)
                else:
                    print(f"Aviso: Não foi possível ler a imagem {src_file}")

            # Se for qualquer outro ficheiro (labels .txt, data.yaml, README)
            else:
                shutil.copy2(src_file, dest_file)

    print("Conversão e cópia da base de dados concluídas com sucesso!")


if __name__ == "__main__":
    main()
