import json  # Importing the JSON module for working with JSON data
import requests  # Importing the Requests module for making HTTP requests
import os  # Importing the OS module for working with files and directories
import re  # Importing the Regular Expression module for pattern matching and string manipulation
import datetime  # Importing the Datetime module for working with dates and times
import ijson  # Importing the ijson module for iterative JSON parsing
import urllib3  # Importing the urllib3 module for working with URLs
import urllib.request  # Importing the urllib.request module for making URL requests
import time
import glob
import cv2
import random
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)  # Disable SSL verification warning

def get_valid_filename(s):
    # Function to sanitize a string and make it a valid filename
    s = str(s).strip().replace(' ', '_')  # Strip leading/trailing whitespaces and replace spaces with underscores
    return re.sub(r'(?u)[^-\w.]', '', s)  # Remove any characters that are not alphanumeric, hyphen, or dot

def checkdir(dir_path):
    # Function to create a directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)

def writefile(url, file_path):
    # Function to download a file from a given URL and save it to the specified file path
    if not os.path.isfile(file_path):  # Check if the file already exists
        r = requests.get(url, verify=False)  # Make an HTTP GET request to the URL (SSL verification disabled)
        with open(file_path, 'wb') as f:
            f.write(r.content)  # Write the response content to the file

def get_card_data_and_download(i, card_name, set_code, card_number):    
    # Function to get card data from the Scryfall API and download the card image
    search_url = "https://api.scryfall.com/cards/search"
    set_code = set_code.replace("(", "").replace(")", "")  # Remove parentheses from the set code.
    
    # Output paths.
    file_path = os.path.join(dl_path, f"{card_name} ({set_code}) {card_number}.png")
    file_path_back = os.path.join(dl_path, f"{card_name} ({set_code}) {card_number} (Back).png")
    
    # Reduce unnecessary downloads.
    if os.path.isfile(file_path):
        if os.path.isfile(file_path_back):
            return [file_path, file_path_back]
        return [file_path]
    
    print(f"Download | {card_name}")
    if set_code and card_number:
        query = f'name:"{card_name}" set:{set_code} number:{card_number}'  # Query string for searching the card
    else:
        query = f'name:"{card_name}"'  # Query string for searching the card without set code and card number
    
    params = {
        "q": query,
        "format": "json"
    }
    response = requests.get(search_url, params=params)  # Make an HTTP GET request to the search URL with the query
    if response.status_code == 200:
        data = response.json()
        if "data" in data and len(data["data"]) > 0:  # Check if card data is present
            card_data = data["data"][0]  # Get the first card data from the response
            result = save_card_image(card_data, file_path, file_path_back)  # Save the card image
            time.sleep(0.5)
            return result
        else:
            print(f"No card found for '{card_name}' in set '{set_code}' with number '{card_number}'")
    else:
        print(f"Error retrieving card data for '{card_name}' in set '{set_code}' with number '{card_number}'. Response: {response.text}")
    
    return []

def save_card_image(card, file_path, file_path_back):
    sfc = False
    mdfc = False

    if 'image_uris' in card:
        writefile(card['image_uris']['png'], file_path)  # Download and save the card image
        return [file_path]
    elif 'type_line' in card and card['type_line'] != 'Card // Card':
        if 'card_faces' in card and len(card['card_faces']) == 2:
            mdfc = True
    elif 'layout' in card and card['layout'] == 'reversible_card':
        if 'card_faces' in card and len(card['card_faces']) == 2:
            mdfc = True
    
    if mdfc:
        writefile(card['card_faces'][0]['image_uris']['png'], file_path)
        writefile(card['card_faces'][1]['image_uris']['png'], file_path_back)
        return [file_path, file_path_back]
    
    print(f"No valid image found for card: {name}")
    return []

def scale_card(path_in, path_out):
    image = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)
    
    # Remove copyright by making it transparent.
    image = remove_copyright(image)
    
    # Crop all transparent pixels.
    mask = np.where(image[:, :, 3] != 255)
    image[mask] = (0, 0, 0, 0)
    
    # Fill transparent space.
    size = np.shape(image)
    for y in range(size[0]):
        for x in range(size[1]):
            if image[y, x, 3] == 0:
                d = [0, 0]
                
                if y < size[0] / 2:
                    d[0] = +1
                else:
                    d[0] = -1
                    
                if x < size[1] / 2:
                    d[1] = +1
                else:
                    d[1] = -1
                
                p = [y, x]
                for i in range(size[0]):
                    p[0] += d[0]
                    p[1] += d[1]
                    if image[p[0], p[1], 3] != 0:
                        image[y, x, :] = image[p[0], p[1], :]
                        break
    
    md = image
    md = cv2.medianBlur(md, 7)
    md = cv2.GaussianBlur(md, [21, 21], 0)
    image[mask] = md[mask]
    
    # Extend border.
    image = cv2.resize(image, dsize=[759, 1057], interpolation=cv2.INTER_CUBIC)
    addy = 34
    addx = 33
    image = cv2.copyMakeBorder(image, addy, addy, addx, addx, cv2.BORDER_REPLICATE)
    image = add_watermark(image)
    
    cv2.imwrite(path_out, image)

def imdif(imb, y1, x1, y2, x2):
    eps = 0.000001
    a = imb[y1, x1, 0:3]
    b = imb[y2, x2, 0:3]
    an = np.linalg.norm(a) + eps
    bn = np.linalg.norm(b) + eps
    return np.linalg.norm(b / bn - a / an) + np.linalg.norm((b - a) / np.max([an, bn]))

def remove_copyright(image):
    mask = image[:, :, 0].copy() * 0
    
    imflt = image.astype(float) / 255.0
    
    ystart = 970
    yend = 1040
    xstart = 435
    xend = 700
    
    # Mask adder.
    for y in range(ystart, 1010):
        for x in range(xstart, xend + 1):            
            p = imflt[y, x, 0:3]
            d = np.dot(p, p)
            if np.dot(d, d) > 0.03 and np.dot(p, [p[1], p[2], p[0]]) > 0.985 * d:
                if np.dot(d, d) > 0.01:
                    r1my = 2
                    r1py = 8
                    r1x = 4
                else:
                    r1my = 0
                    r1py = 0
                    r1x = 0
                
                for dy in range(-r1my, +r1py + 1):
                    for dx in range(-r1x, +r1x + 1):
                        mask[y + dy, x + dx] = 1
    
    # Mask remover.
    for y in range(ystart, 1010):
        for x in range(xstart, xend + 1):
            pr = imflt[(y - 3):(y + 1), x, 0:3]
            for dy in range(-4, 0):
                p = pr[dy + 4, :]
                d = np.dot(p, p)
                # Very bright saturation directly above.
                if np.dot(d, d) > 0.1 and np.dot(p, [p[1], p[2], p[0]]) < 0.9 * d:
                    mask[y, x] = 0
                    break
    
    # P/T box.
    yt = 945
    yb = 965
    xl = 600
    xr = 680
    imb = cv2.blur(imflt, [3, 5], 0)
    bb = np.zeros([yb - yt])
    for i in range(0, yb - yt):
        b02 = imdif(imb, yt + i, xl, yt + i + 1, xl)
        b13 = imdif(imb, yt + i, xr, yt + i + 1, xr)
        bb[i] = b02 + b13
    bv = np.max(bb) - np.median(bb)
    iscreature = bv < 0.08
    if iscreature:
        mask[920:985, 570:705] = 0
    
    # Now fill in according to mask.
    for y in reversed(range(ystart, yend)):
        for x in range(xstart - 4, xend + 4 + 1):
            if mask[y, x] == 1:
                dy = 1
                while mask[y + dy, x] == 1:
                    dy += 1
                random.seed(100 * (745 * y + x) + dy)
                dx = random.randint(0, 16) - 8
                image[y, x, 0:3] = image[y + dy, x + dx, 0:3]
                mask[y, x] = 0
    
    return image

def copy_override(path_in, path_out):
    image = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)
    image = cv2.resize(image, dsize=[825, 1125], interpolation=cv2.INTER_AREA)
    image = add_watermark(image)
    cv2.imwrite(path_out, image)

def add_watermark(image):
    imflt = image.astype(float)
    imflt[wmy1:wmy2, wmx1:wmx2, 0] *= wmainv
    imflt[wmy1:wmy2, wmx1:wmx2, 1] *= wmainv
    imflt[wmy1:wmy2, wmx1:wmx2, 2] *= wmainv
    imflt[wmy1:wmy2, wmx1:wmx2, 0:3] += wmrgb
    return imflt.astype('uint8')

def download_cards_list():
    # Function to download card images from a list of cards
    with open("cards.txt", "r") as file:
        card_list = file.readlines()  # Read the list of cards from a file

    i = 0
    for card in card_list:
        i += 1
        card_data = card.strip().split(" ")  # Split the card data into card name, set code, and card number
        if card_data[0].isdigit():
            del card_data[0]
        while card_data[-1] == '#print' or card_data[-1] == '*F*':
            del card_data[-1]

        if len(card_data) < 3:
            card_name = card.strip()
            set_code = ""
            card_number = ""
        else:
            card_name = " ".join(card_data[:-2])
            set_code = card_data[-2]
            card_number = card_data[-1]

        # Remove MDFC back-side from names.
        card_name = card_name.split("/")[0].strip()
        
        # Output targets.
        out_png = os.path.join(out_path, f"{i:03d} {card_name}.png")
        out_png_back = os.path.join(out_path, f"{i:03d} {card_name} (Back).png")
        
        if os.path.exists(out_png):
            print(f"Skipping | {card_name}")
            continue
        
        # Check for overrides.
        or_jpg = os.path.join(or_path, f"{card_name}.jpg")
        or_png = os.path.join(or_path, f"{card_name}.png")
        or_jpg_exists = os.path.isfile(or_jpg)
        or_png_exists = os.path.isfile(or_png)
        
        if or_jpg_exists or or_png_exists:
            print(f"Proxy    | {card_name}")
            # Copy front.
            if or_jpg_exists:
                copy_override(or_jpg, out_png)
            elif or_png_exists:
                copy_override(or_png, out_png)
            
            # Copy back.
            or_jpg_back = os.path.join(or_path, f"{card_name} (Back).jpg")
            or_png_back = os.path.join(or_path, f"{card_name} (Back).png")
            if os.path.isfile(or_jpg_back):
                copy_override(or_jpg_back, out_png_back)
            elif os.path.isfile(or_png_back):
                copy_override(or_png_back, out_png_back)
        else:
            print(f"Scryfall | {card_name}")
            files = get_card_data_and_download(i, card_name, set_code, card_number)
            files_len = len(files)
            if files_len > 0:
                scale_card(files[0], out_png)
                if files_len > 1:
                    scale_card(files[1], out_png_back)
    
    for path in list(glob.glob(os.path.join(tk_path, "*"))):
        file = os.path.basename(path)
        file_split = file.split(".")
        name = file_split[0].strip()
        if os.path.isfile(path) and not name.endswith(" (Back)"):
            i += 1
            print(f"Token    | {name}")
            out_png = os.path.join(out_path, f"{i:03d} {name}.png")
            copy_override(path, out_png)
            
            tk_jpg_back = os.path.join(tk_path, f"{name} (Back).jpg")
            tk_png_back = os.path.join(tk_path, f"{name} (Back).png")
            out_png_back = os.path.join(out_path, f"{i:03d} {name} (Back).png")
            if os.path.isfile(tk_jpg_back):
                copy_override(tk_jpg_back, out_png_back)
            elif os.path.isfile(tk_png_back):
                copy_override(tk_png_back, out_png_back)
            else:
                print(f"Token {name} lacking back face")

# Output directory for card images
output_dir = os.path.join(os.getcwd(), "art")
print("Writing files to", output_dir)

dl_path = os.path.join(output_dir, "Scryfall")
or_path = os.path.join(output_dir, "Override")
tk_path = os.path.join(output_dir, "Token")
out_path = os.path.join(output_dir, "Out")

checkdir(dl_path)
checkdir(or_path)
checkdir(out_path)

wm = cv2.imread(os.path.join(os.getcwd(), "wm", "wm1.png"), cv2.IMREAD_UNCHANGED).astype(float)
wmy1 = 1039
wmx1 = 555
wmy2 = wmy1 + wm.shape[0]
wmx2 = wmx1 + wm.shape[1]
wma = wm[:, :, 3] / 255
wmainv = 1 - wma
wmrgb = wm[:, :, 0:3]
wmrgb[:, :, 0] *= wma
wmrgb[:, :, 1] *= wma
wmrgb[:, :, 2] *= wma

download_cards_list()
