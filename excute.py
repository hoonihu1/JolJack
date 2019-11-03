import subprocess
import os
import tornado.httpserver, tornado.ioloop, tornado.options, tornado.web, os.path, random, string
from tornado.options import define, options
import cv2, glob, dlib

define("port", default=9000, help="run on the given port", type=int)

class Application(tornado.web.Application):

    def __init__(self):
        handlers = [
            (r"/", IndexHandler),
            (r"/upload", UploadHandler),
        ]
        settings = {
            "static_path": os.path.join(os.path.dirname(__file__), "view"),
            "static_url_prefix": "/view/",
        }
        tornado.web.Application.__init__(self, handlers, **settings)

class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("upload_form.html")

class UploadHandler(tornado.web.RequestHandler):

    final_filename = ""
    input_tensor_process = "label_image.py --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt --input_layer=Placeholder --output_layer=final_result --image=uploads/"

    def post(self):
        age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        file1 = self.request.files['file1'][0]
        original_fname = file1['filename']
        extension = os.path.splitext(original_fname)[1]
        fname = ''.join(random.choice(string.ascii_lowercase + string.digits) for x in range(6))
        self.final_filename = fname+extension
        output_file = open("uploads/" + self.final_filename, 'wb')
        output_file.write(file1['body'])
        learnpimpe = self.learn_machine()
        learn_age_sex = self.learn_machine_age_and_sex()
        learn_age_sex = list(learn_age_sex)
        sex = learn_age_sex[0]
        age = learn_age_sex[1]

        # for i in range(age_list):
        #     if age == i:
        print(learnpimpe)
        print(learn_age_sex[0])
        self.get()
        # self.write("<img src=""uploads/"+self.final_filename+">"+"<br>")
        # self.write(learnpimpe+"<br>")
        # self.write(str(learn_age_sex))

    def get(self):
        self.render("view/index.html")

    def learn_machine(self):
        proc = subprocess.Popen(
            self.input_tensor_process+self.final_filename,
            stdout=subprocess.PIPE, shell=True)
        (out, err) = proc.communicate()
        out = out.decode('utf-8')
        return out
        # self.write(out)

    def learn_machine_age_and_sex(self):
        age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
        gender_list = ['Male', 'Female']
        detector = dlib.get_frontal_face_detector()

        age_net = cv2.dnn.readNetFromCaffe(
            'models/deploy_age.prototxt',
            'models/age_net.caffemodel')
        gender_net = cv2.dnn.readNetFromCaffe(
            'models/deploy_gender.prototxt',
            'models/gender_net.caffemodel')

        img_list = glob.glob('uploads/'+self.final_filename)

        for img_path in img_list:
            img = cv2.imread(img_path)

            faces = detector(img)

            for face in faces:
                x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()

                face_img = img[y1:y2, x1:x2].copy()

                blob = cv2.dnn.blobFromImage(face_img, scalefactor=1, size=(227, 227),
                                             mean=(78.4263377603, 87.7689143744, 114.895847746),
                                             swapRB=False, crop=False)

                # predict gender
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]

                # predict age
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]

                print(gender, age)

                return gender, age

def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()