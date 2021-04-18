import os
import shutil

print("Making sub-directories...\n")

def createSubDatasets(directory):
    for f in os.listdir(directory):
        for sf in os.listdir(f"{directory}/{str(f)}"):
            for ssf in os.listdir(f"{directory}/{str(f)}/{sf}/train"):
                for imageName in os.listdir(f"{directory}/{str(f)}/{sf}/train/{ssf}"):
                    folderName = imageName[:7]
                    folder_dir = f"{directory}/{str(f)}/{sf}/train/{ssf}/{folderName}"
                    if not os.path.exists(folder_dir):
                        os.mkdir(folder_dir)
                        os.chmod(folder_dir, 0o444)
                        shutil.copy(os.path.join(
                            f"{directory}/{str(f)}/{sf}/train/{ssf}/{imageName}"), f"{folder_dir}/{imageName}")
                    else:
                        shutil.copy(os.path.join(
                            f"{directory}/{str(f)}/{sf}/train/{ssf}/{imageName}"), f"{folder_dir}/{imageName}")

                print(f"Set {ssf} is completed.")
