import gdown
def gdownload(file_id, output_name):
    google_path = 'https://drive.google.com/uc?id='
    gdown.download(google_path+file_id,output_name,quiet=False)