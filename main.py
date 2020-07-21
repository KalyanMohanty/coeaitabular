from flask import Flask, request, render_template, send_from_directory, jsonify, Response, url_for, redirect,send_file
from werkzeug.datastructures import FileStorage
from openpyxl import load_workbook
from google.cloud import storage
import urllib.request
from io import StringIO
from werkzeug.utils import secure_filename
import os

try:
    from PIL import Image
except ImportError:
    import Image
import logging
#GOOGLE_APPLICATION_CREDENTIALS = "satwik-credentials.json"
#client = storage.Client.from_service_account_json('satwik-credentials.json')
client = storage.Client()
app = Flask(__name__)
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
#UPLOAD_FOLDER = 'static/'
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
#fs = FileStorage()

@app.route("/")
def index():
    return render_template("tabular.html")


@app.route('/tabular', methods=['POST'])
def tabular():
    """Process the uploaded file and upload it to Google Cloud Storage."""
    FileStorage.tell() != 0
    uploaded_file = request.files.get('file')
    if request.method == 'POST':
        if not uploaded_file:
            return 'No file uploaded.', 400

        # Open Image with Pillow
        else:
            image = Image.open(uploaded_file)

            # Resize image with Pillow (Problem still occurs without this step)
            image.resize((300, 300))

            # Create Filestorage object (This is the wrapper Flask uses for their file uploads)

            # Save image with Pillow into FileStorage object.
            image.save(FileStorage, format='JPEG')

            # Verify that image was resized and works
            Image.open(FileStorage).show()

            # Create a Cloud Storage client.
            gcs = storage.Client()

            # Get the bucket that the file will be uploaded to.
            bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)

            # Create a new blob and upload the file's content.
            blob = bucket.blob(uploaded_file.filename)
            FileStorage.seek(0)
            blob.upload_from_file(FileStorage)

            # The public URL can be used to directly access the uploaded file via HTTP.
            url_name = blob.public_url
            return redirect('/downloadfile/' + url_name)
    return render_template('tabular.html')


# Download API
@app.route("/downloadfile/<url_name>", methods=['GET'])
def download_file(url_name):
    return render_template('download.html', value=url_name)


@app.route('/return-files/<filename>')
def return_files_tut(filename):
    file_path = UPLOAD_FOLDER + filename
    return send_file(file_path, as_attachment=True, attachment_filename='')


@app.route("/extract/<url_name>", methods=["GET", "POST"])
def extract(url_name):
    # filename = request.form['image']
    #  target = os.path.join(APP_ROOT, 'blob/')
    # destination = "/".join([target, filename])
    # image_path = os.path.join(app.config['static/'], filename)
    req = urllib.request.Request("{url_name}")
    image = StringIO.StringIO(urllib.request.urlopen(req).read())
    import cv2
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    import csv

    import pytesseract

    # read your file
    # file = upload()
    # file = send_image(file)
    # image = cv2.imread(destination, 0)
    # image.shape
    # thresholding the image to a binary image
    # thresh,img_binary = cv2.threshold(image,120,255,cv2.THRESH_BINARY |cv2.THRESH_OTSU)#inverting the image
    img_binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img_binary = 255 - img_binary  # inverting  the image
    # cv2.imwrite('cv_inverted.png', img_binary)  # Plotting the image to see the output
    plotting = plt.imshow(img_binary, cmap='gray')
    plt.show()

    # In[3]:

    # np.array(image).shape[0]

    # In[4]:

    #  np.array(image).shape[1]

    # In[5]:

    Image_size = 2000
    # length_x,width_y = image.size
    length_x = np.array(image).shape[1]
    width_y = np.array(image).shape[0]
    factor = max(1, int(Image_size // length_x))
    size = factor * length_x, factor * width_y

    # In[6]:

    # In[42]:

    # Length of kernel as 100th of total width
    kernel_length = 5
    # ernel_length = factor
    # Defining a vertical kernel to detect all vertical lines of image
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # Defining a horizontal kernel to detect all horizontal lines of image
    horizantal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # In[43]:

    image_1 = cv2.erode(img_binary, vertical_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, vertical_kernel, iterations=3)
    # cv2.imwrite("testing-vertical.jpg", vertical_lines)  # Plotting the generated image
    plotting = plt.imshow(image_1, cmap='gray')
    plt.show()

    # In[44]:

    # Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_binary, horizantal_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, horizantal_kernel, iterations=3)
    #  cv2.imwrite("horizontal.jpg", horizontal_lines)  # Plotting the generated image
    plotting = plt.imshow(image_2, cmap='gray')
    plt.show()

    # In[45]:

    # addweighted weighs horizantal and horizantal lines the same
    # bitwise or and bitwise_not for exclusive or and not operations

    # In[47]:

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)  # Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh, 120, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # threshold binary or otsus binarization(mainly for bimodal)
    # thresh is thresholded value used,next is the thresholded image
    # cv2.imwrite("img_combined.jpg", img_vh)
    bitxor = cv2.bitwise_xor(image, img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    # cv2.imwrite("bitnot.jpg", bitnot)
    # Plotting the generated image
    plotting = plt.imshow(bitnot, cmap='gray')
    plt.show()

    # #
    # The mode cv2.RETR_TREE finds all the promising contour lines and reconstruct
    # s a full hierarchy of nested contours. The method cv2.CHAIN_APPROX_SIMPLE
    # returns only the endpoints that are necessary for drawing the contour line.

    # In[12]:

    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # In[13]:

    import numpy as np
    import argparse
    import imutils
    import cv2

    # In[14]:

    # get_ipython().system('pip install imutils')

    # In[15]:

    def sort_contours(cnts, method="left-to-right"):  # initialize the reverse flag and sort index
        reverse = False
        i = 0  # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True  # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1  # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                            key=lambda b: b[1][i],
                                            reverse=reverse))  # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # ###following a top-down approach for sorting contours

    # In[16]:

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, "top-to-bottom")

    # In[17]:

    # Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]  # Get mean of heights
    mean = np.mean(heights)

    #

    # In[51]:

    box = []  # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (h < 1000 and w < 1000):
            fimage = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            box.append([x, y, w, h])
    plotting = plt.imshow(fimage, cmap='gray')
    plt.show()

    # In[19]:

    # len(box)

    # In[20]:

    print(box)

    # In[21]:

    # len(box)

    # In[22]:

    # classifying into rows and columns

    # In[23]:

    row = []
    column = []
    j = 0

    for i in range(len(box)):

        if (i == 0):
            column.append(box[i])
            previous = box[i]

        else:
            if (box[i][1] <= previous[1] + mean / 2):
                column.append(box[i])
                previous = box[i]

                if (i == len(box) - 1):
                    row.append(column)

            else:
                row.append(column)
                column = []
                previous = box[i]
                column.append(box[i])

    print(column)
    print(row)

    # In[24]:

    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol
    print(countcol)

    # In[ ]:

    # In[25]:

    # row[0]

    # In[ ]:

    # In[26]:

    count = 0
    for i in range(len(row)):
        count += 1
        print(row[i])
    print(count)

    # In[27]:

    # Retrieving the centers and sorting them
    center = [int(row[i][j][0] + row[i][j][2] / 2) for j in range(len(row[i])) if row[0]]
    center = np.array(center)
    center.sort()

    # In[28]:

    # print(center)

    # In[29]:

    # Regarding the distance to the columns center, the boxes are arranged in respective order
    finalboxes = []
    for i in range(len(row)):
        lis = []
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center - (row[i][j][0] + row[i][j][2] / 4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)

    # In[30]:

    # finalboxes

    #
    # #psm:Set Tesseract to only run a subset of layout analysis and assume a certain form of imag
    #
    #
    # 0 = Orientation and script detection (OSD) only.
    # 1 = Automatic page segmentation with OSD.
    # 2 = Automatic page segmentation, but no OSD, or OCR. (not implemented)
    # 3 = Fully aut1matic page segmentation, but no OSD. (Default)
    # 4 = Assume a single column of text of variable sizes.
    # 5 = Assume a single uniform block of vertically aligned text.
    # 6 = Assume a single uniform block of text.
    # 7 = Treat the image as a single text line.
    # 8 = Treat the image as a single word.
    # 9 = Treat the image as a single word in a circle.
    # 10 = Treat the image as a single character.
    # 11 = Sparse text. Find as much text as possible in no particular order.
    # 12 = Sparse text with OSD.
    # 13 = Raw line. Treat the image as a single text line,
    #      bypassing hacks that are Tesseract-specific.

    # #
    #     Specify OCR Engine mode. The options for N are:
    #
    #     0 = Original Tesseract only.
    #     1 = Neural nets LSTM only.
    #     2 = Tesseract + LSTM.
    #     3 = Default, based on what is available.
    #
    #

    # In[ ]:

    # custom_config = r'--oem 3 --psm 6 outputbase digits'

    # In[57]:

    # from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer = []

    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner = ''
            if (len(finalboxes[i][j]) == 0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y, x, w, h = finalboxes[i][j][k][0], finalboxes[i][j][k][1], finalboxes[i][j][k][2], \
                                 finalboxes[i][j][k][3]
                    finalling = bitnot[x:x + h, y:y + w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalling, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=[0, 255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel, iterations=1)  ##size of foreground object increases
                    erosion = cv2.erode(dilation, kernel, iterations=2)  # Thickness of foreground object decreases
                    out = pytesseract.image_to_string(finalling)
                    if (len(out) == 0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3 --oem 3 -c '
                                                                          'tessedit_char_whitelist=0123456789')
                    inner = inner + " " + out
                outer.append(inner)
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    print(dataframe)
    data = dataframe.style.set_properties(align="left")
    return Response(
        dataframe.to_csv(),
        mimetype="text/csv",
        headers={"Content-disposition":
                     "attachment; filename=filename.csv"})


# print(data)
# return Response(dataframe.to_json(orient="columns"), mimetype='application/json')
# return render_template('table.html', tables=[dataframe.to_html(classes='data')], titles=dataframe.columns.values)

# In[55]:


#  data.to_excel("temp.xlsx")
# file_name = 'document_template.xltx'
#  wb = load_workbook('temp.xlsx')
# wb.save(file_name, as_template=True)
# return send_from_directory(file_name, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)