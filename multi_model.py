import os
from flask import Flask, render_template, request, flash, make_response, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import pytesseract


app = Flask(__name__)
app.config['SECRET_KEY'] = '123456'

#允许上传type
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', "PNG", "JPG", 'JPEG', 'mp4'])  # 大写的.JPG是不允许的

#check type
def allowed_file(filename):
    return '.' in filename and filename.split('.', 1)[1] in ALLOWED_EXTENSIONS
    # 圆括号中的1是分割次数

#upload path
UPLOAD_FOLDER = './uploads'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """"目前只支持上传英文名"""
    if request.method == 'POST':
        #获取上传文件
        file = request.files['file']
        print(dir(file))
        #检查文件对象是否存在且合法
        if file and allowed_file(file.filename):  # 规定file都有什么属性
            filename = secure_filename(file.filename)  # 把汉字文件名抹掉了，所以下面多一道检查
            if filename != file.filename:
               flash("only support ASCII name")
               return render_template('upload.html')
            #save
            try:
                file.save(os.path.join(UPLOAD_FOLDER, filename))  # 现在似乎不会出现重复上传同名文件的问题
            except FileNotFoundError:
                os.mkdir(UPLOAD_FOLDER)
                file.save(os.path.join(UPLOAD_FOLDER, filename))
            # return redirect(url_for('update', fileName=filename))

            # 读取图像
            image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
            # 转换为灰度图像
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # 使用 Tesseract 进行 OCR
            text = pytesseract.image_to_string(gray_image, config='--psm 6')
            # 输出识别的文本
            # print("识别的文本:", text)

            return redirect(url_for('upload_file', fileName=filename)), text
        else:
            return 'Upload Failed'
    else: #GET方法
        return render_template('upload.html')



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1234, debug=True)

