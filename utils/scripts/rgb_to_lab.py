import cv2
import os
import shutil
from pathlib import Path

def main():
    # Caminhos da base original e da nova base convertida
    input_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26")
    output_base = Path("/home/antoniovinicius/projects/sandbox_yolo26/datasets/isic_2018_task1_yolo26_lab")

    # Extensões de imagem que queremos converter
    image_extensions = {'.jpg', '.jpeg', '.png'}

    print(f"Iniciando conversão para LAB...\nOrigem: {input_base}\nDestino: {output_base}\n")

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
                    # 2. Converte de BGR para LAB
                    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                    
                    # 3. Guarda a imagem convertida na nova pasta
                    cv2.imwrite(str(dest_file), img_lab)
                else:
                    print(f"Aviso: Não foi possível ler a imagem {src_file}")
            
            # Se for qualquer outro ficheiro (labels .txt, data.yaml, README)
            else:
                shutil.copy2(src_file, dest_file)

    print("Conversão e cópia da base de dados concluídas com sucesso!")

if __name__ == "__main__":
    main()