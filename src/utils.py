import os
import shutil

def edit_test_folder(path):
    test_folder = path
    for item in os.listdir(test_folder):
        item_path = os.path.join(test_folder, item)
        # Check if the item is a folder and starts with "test_"
        if os.path.isdir(item_path) and item.startswith("test_"):
            # Look for PNG files inside the folder
            for file_name in os.listdir(item_path):
                if file_name.endswith(".png"):
                    # Move the PNG file to the 'test' folder
                    src_path = os.path.join(item_path, file_name)
                    dest_path = os.path.join(test_folder, file_name)
                    shutil.move(src_path, dest_path)
            
            # Delete the folder after moving the files
            shutil.rmtree(item_path)
    print("All images moved and unnecessary folders deleted.")
    return None

edit_test_folder("test")