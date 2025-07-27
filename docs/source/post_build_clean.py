
import shutil
import glob
import os

def delete_tuto_after_build():
    base_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        "_static", "supplementary_material", "tuto"
    ))
    if not os.path.isdir(base_path):
        print(f"Dossier introuvable : {base_path}")
        return
    for ext in ("*.html", "*.md", "*.hdf5"):
        for file_path in glob.glob(os.path.join(base_path, ext)):
            if os.path.isfile(file_path):
                print(f"Suppression du fichier : {file_path}")
                os.remove(file_path)
    for dir_path in glob.glob(os.path.join(base_path, "started_*")):
        if os.path.isdir(dir_path):
            print(f"Suppression du dossier : {dir_path}")
            shutil.rmtree(dir_path)


if __name__ == "__main__":
    delete_tuto_after_build()