import os
import shutil
from glob import glob
from pathlib import Path

shutil.rmtree("matches", ignore_errors=True)
os.makedirs("matches", exist_ok=True)
if os.path.exists("matches.txt"):
    os.remove("matches.txt")

import cv2
import numpy as np

def clean(path):
    path = path.replace('$','')
    path = path.replace('\r','')
    return path

def purge(paths):
    for p in paths:
        p_ = clean(p)
        if p == p_:
            continue
        os.rename(p, p_)

# Fix Paths
# regex_generated_panoramas = 'generated_panorama/hdr/*'
# paths_generated_panoramas = glob(regex_generated_panoramas)
# purge(paths_generated_panoramas)
# regex_generated_panoramas = 'generated_panorama/holistic/*'
# paths_generated_panoramas = glob(regex_generated_panoramas)
# purge(paths_generated_panoramas)

# regex_generated_panoramas = 'generated_panorama/ldr/*'
# paths_generated_panoramas = glob(regex_generated_panoramas)
# purge(paths_generated_panoramas)

regex_generated_panoramas = 'generated_panorama/ldr/hrldr_*.png'
paths_generated_panoramas = glob(regex_generated_panoramas)
# paths_generated_panoramas = paths_generated_panoramas[:10]
print(f"Found {len(paths_generated_panoramas)} generated panoramas")

# regex_outdoorPanos = 'outdoorPanosExr/*.exr'
regex_outdoorPanos = 'outdoorPanosEx/*.exr' #typo
# regex_outdoorPanos = 'generated_panorama/ldr/ldr_*.png'
paths_outdoorPanos = glob(regex_outdoorPanos)
# paths_outdoorPanos = paths_outdoorPanos[:10]
print(f"Found {len(paths_outdoorPanos)} outdoor panoramas")


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
def load_image(path):
    # load the image
    img = cv2.imread(path, flags = cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if path.endswith('.exr'):
        img = img * np.power(2, 6)
        img = np.power(img, 1.0/2.2)
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

    while img.shape[0] > 1000:
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)

    return img


# Initiate ORB detector
orb = cv2.ORB_create()

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(
    algorithm = FLANN_INDEX_LSH,
    table_number = 6, # 12
    key_size = 12, # 20
    multi_probe_level = 1
) #2
search_params = dict(checks=50) # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

def get_keypoints_descriptors(p_img):
    # find the keypoints and descriptors with ORB
    img = load_image(p_img)
    _, des = orb.detectAndCompute(img,None)
    return {"des":des}

dict_generated_panoramas = {}
for p_generated in paths_generated_panoramas:
    dict_generated_panoramas[p_generated] = get_keypoints_descriptors(p_generated)

dict_outdoor_panoramas = {}
for p_outdoor in paths_outdoorPanos:
    dict_outdoor_panoramas[p_outdoor] = get_keypoints_descriptors(p_outdoor)


# Find matches
thresh_matches = 5
min_num_nearest_neighbors = 2
for p_outdoor in paths_outdoorPanos:
    best_match_count=10
    best_match = None
    print("Querying match for: ", p_generated)
    # des1 = dict_generated_panoramas[p_generated]["des"]
    des1 = dict_outdoor_panoramas[p_outdoor]["des"]
    if des1 is not None and len(des1)>= min_num_nearest_neighbors:
        for p_generated in paths_generated_panoramas:
            # des2 = dict_outdoor_panoramas[p_outdoor]["des"]
            des2 = dict_generated_panoramas[p_generated]["des"]
            if des2 is not None and len(des2) >= min_num_nearest_neighbors:
                # Get matches
                matches = flann.knnMatch(des1, des2,k=2)

                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]

                if len(matches) >= thresh_matches:
                    count=0
                    for i,d in enumerate(matches):
                        if isinstance(d, list) and len(d) >= 2:
                            m,n = d
                            if m.distance < 0.7*n.distance:
                                matchesMask[i]=[1,0]
                                count += 1
                    if count > best_match_count:
                        print(f"\tNew best match (count={count})")
                        best_match_count = count
                        best_match = p_outdoor

        # Draw best match
        if best_match:
            img1 = load_image(p_generated)
            kp1, des1 = orb.detectAndCompute(img1,None)
            img2 = load_image(best_match)
            kp2, des2 = orb.detectAndCompute(img2,None)
            matches = flann.knnMatch(des1,des2,k=2)
            matchesMask = [[0,0] for i in range(len(matches))]
            for i,d in enumerate(matches):
                if isinstance(d, list) and len(d) >= 2:
                    m,n = d
                    if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]

            draw_params = dict(
                matchColor = (0,255,0),
                singlePointColor = (255,0,0),
                matchesMask = matchesMask,
                flags = cv2.DrawMatchesFlags_DEFAULT
            )
            img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

            path_out = Path("./matches/") / (Path(p_generated).stem+"_matched_"+Path(best_match).stem+"_.png")
            img3 = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path_out.as_posix(), img3)

            with open("matches.txt", "a") as f:
                f.write(f"{best_match_count}, {Path(p_generated).stem}, {Path(best_match).stem}\n")

        else:
            print(f"\tNo match found for: {p_generated}")
    else:
        print(f"\tNo keypoints found for: {p_generated}")