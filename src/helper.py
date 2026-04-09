'''
Author: Armand Meijers
Date: 06/04/2026
Description: holds all helper functions 
'''

#imports
import os

#checks if folder/ file exists and if not creates it
def path_checker_creator(path, exist_ok=True):
    """
    Checks if path passed in exsits for both folers and files

    Args:
        path (str): path of file you want to check
    """

    try:
        root, ext = os.path.splitext(path)

        # for FILE
        if ext:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                open(path, "w").close()
        
        # for DIRECTORY
        else:
            os.makedirs(path, exist_ok=True)
            
        print(f"[LOG] Created {path}")

    #erro log
    except Exception as e:
        print(f"[ERROR] {e} | error checking path")
        return False

    return True