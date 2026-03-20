"""
Script para limpar datasets de segmentação YOLO.
Ele varre a pasta de labels e, se encontrar alguma anotação no formato
de Bounding Box (exatamente 5 valores), deleta o .txt e a imagem correspondente.
"""

import os
import glob

# Constante com as extensões de imagens suportadas
IMAGE_EXTENSIONS = ('.jpg', '.png', '.jpeg', '.JPG')


def remove_invalid_labels(label_dir: str, image_dir: str) -> None:
    """
    Inspeciona os arquivos de label e remove pares (label + imagem)
    que não estejam no formato de polígono (segmentação).
    """
    deleted_count = 0
    
    # Busca todos os arquivos .txt na pasta de labels
    txt_files = glob.glob(os.path.join(label_dir, "*.txt"))

    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        should_delete = False
        
        for line in lines:
            values = line.strip().split()
            
            # Se a linha tiver exatamente 5 valores (classe, x, y, w, h), é Bounding Box!
            # Para segmentação, o esperado são dezenas de valores (polígonos).
            if len(values) == 5:
                should_delete = True
                break
                
        if should_delete:
            # Apaga o arquivo de label (.txt)
            os.remove(txt_file)
            
            # Extrai o nome base do arquivo (sem a extensão e sem o caminho)
            base_name = os.path.splitext(os.path.basename(txt_file))[0]
            
            # Procura a imagem correspondente e a deleta
            for ext in IMAGE_EXTENSIONS:
                img_path = os.path.join(image_dir, base_name + ext)
                if os.path.exists(img_path):
                    os.remove(img_path)
                    break # Interrompe o loop pois já encontrou e deletou a imagem
                    
            deleted_count += 1
            print(f"Removido par problemático: {base_name}")

    print(f"\nLimpeza concluída! {deleted_count} arquivos foram apagados.")


def main() -> None:

    label_directory = "datasets/isic_2018_task1_yolo26/train/labels"
    image_directory = "datasets/isic_2018_task1_yolo26/train/images"
    
    print("Iniciando a varredura do dataset em busca de Bounding Boxes...\n")
    remove_invalid_labels(label_directory, image_directory)


# Garante que o script só execute se chamado diretamente
if __name__ == "__main__":
    main()

# Removido par problemático: ISIC_0014833_jpg.rf.6e13ae48f44299c82717e5c9db1cf54e
# Removido par problemático: ISIC_0015216_jpg.rf.6cdfb7ade14de92ba679b4fd46aac6d7
# Removido par problemático: ISIC_0004337_jpg.rf.f1d77fd17316045a74d7af093a58c8d8
# Removido par problemático: ISIC_0015078_jpg.rf.e70c3379f35ea5727231564dc281a504
# Removido par problemático: ISIC_0013196_jpg.rf.2d8f42d40e48be199d21a86a664ec342
# Removido par problemático: ISIC_0015020_jpg.rf.46d42440c0f404634a09345a23cb0a62
# Removido par problemático: ISIC_0004346_jpg.rf.828c20dd3782ca643cdfb0ae23120b3b
# Removido par problemático: ISIC_0015559_jpg.rf.9a0cd2a4af23e4e46ff9e2b4d1f3a1a0

# Limpeza concluída! 8 arquivos foram apagados.