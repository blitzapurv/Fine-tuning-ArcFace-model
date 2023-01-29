import os, time
from deepface.commons import functions, realtime, distance as dst
import warnings
warnings.filterwarnings("ignore")


# extract faces
def face_extract(img_path):
    img_objs = functions.extract_faces(
                img = img_path, 
                target_size = (112,112), 
                detector_backend = 'dlib', 
                grayscale = False,
                enforce_detection = False, 
                align = True)
    return img_objs[0][0].reshape((112,112,3))



# verify if images are same
def pair_verify(img1_path, img2_path, selected_model = None, model_name = 'ArcFace', detector_backend = 'opencv', distance_metric = 'cosine', enforce_detection = False, align = True, normalization = 'base'):
    tic = time.time()
    
    #--------------------------------
    
    target_size = functions.find_target_size(model_name=model_name)
    
    # img pairs might have many faces
    img1_objs = functions.extract_faces(
		img = img1_path, 
		target_size = target_size, 
		detector_backend = detector_backend, 
		grayscale = False, 
		enforce_detection = enforce_detection, 
		align = align)
    
    img2_objs = functions.extract_faces(
		img = img2_path, 
		target_size = target_size, 
		detector_backend = detector_backend, 
		grayscale = False, 
		enforce_detection = enforce_detection, 
		align = align)
    #--------------------------------
    #distances = []
    #regions = []
    # now we will find the face pair with minimum distance
    
    img1_pred = selected_model(img1_objs[0][0])
    img1_pred = img1_pred.numpy().reshape(-1)
    img2_pred = selected_model(img2_objs[0][0])
    img2_pred = img2_pred.numpy().reshape(-1)

    img1_representation = img1_pred #img1_embedding_obj[0]["embedding"]
    img2_representation = img2_pred #img2_embedding_obj[0]["embedding"]
    
    if distance_metric == 'cosine':
        distance = dst.findCosineDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean':
        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
    elif distance_metric == 'euclidean_l2':
        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))
    else:
        raise ValueError("Invalid distance_metric passed - ", distance_metric)
    
    #distances.append(distance)
    #regions.append((img1_region, img2_region))
    
    # -------------------------------
    threshold = dst.findThreshold(model_name, distance_metric)
    #distance = min(distances) #best distance
	#facial_areas = regions[np.argmin(distances)]
    
    toc = time.time()
    
    resp_obj = {
		"verified": True if distance <= threshold else False
		, "distance": distance
		, "threshold": threshold
		, "model": model_name
		, "detector_backend": detector_backend
		, "similarity_metric": distance_metric
		#, "facial_areas": {"img1": facial_areas[0],"img2": facial_areas[1]}
		, "time": round(toc - tic, 2)
	}
    
    return resp_obj