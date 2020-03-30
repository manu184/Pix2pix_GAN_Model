import os
import shutil
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from itertools import islice
from util import html

from flask import Flask, render_template, request, send_from_directory
from flask import url_for
#app = Flask(__name__, static_url_path='/results/edges2handbags/val')
#app = Flask(__name__, static_url_path='/static')
app = Flask(__name__)

def test(number):
    # Variables for Options
    RESULTS_DIR='./results/edges2handbags_2'
    CLASS='edges2handbags'
    DIRECTION='AtoB' # from domain A to domain B
    LOAD_SIZE=256 # scale images to this size
    CROP_SIZE=256 # then crop to this size
    INPUT_NC=1  # number of channels in the input image

    # misc
    GPU_ID=-1   # gpu id
    NUM_TEST=1 # number of input images duirng test
    NUM_SAMPLES=20 # number of samples per input images

    # options
    opt = TestOptions().parse()
    opt.num_threads = 1   # test code only supports num_threads=1
    opt.batch_size = 1   # test code only supports batch_size=1
    opt.serial_batches = True  # no shuffle
    # Options Added
    opt.dataroot = './datasets/edges2handbags'
    opt.results_dir = RESULTS_DIR
    opt.checkpoints_dir = './pretrained_models/'
    opt.name =  CLASS
    opt.direction =  DIRECTION
    opt.load_size = LOAD_SIZE
    opt.crop_size = CROP_SIZE
    opt.input_nc  = INPUT_NC
    opt.num_test  = NUM_TEST
    opt.n_samples = NUM_SAMPLES

    # create dataset
    dataset = create_dataset(opt)
    model = create_model(opt)
    model.setup(opt)
    model.eval()
    print('Loading model %s' % opt.model)
    print(dataset)

    # create website
    web_dir = os.path.join(opt.results_dir, opt.phase + '_sync' if opt.sync else opt.phase)
    webpage = html.HTML(web_dir, 'Training = %s, Phase = %s, Class =%s' % (opt.name, opt.phase, opt.name))

    # sample random z
    if opt.sync:
        z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)

    # test stage
    #for i, data in enumerate(islice(dataset, opt.num_test)):
    for i, data in enumerate(islice(dataset, number, number + opt.num_test)):
        print(data)
        model.set_input(data)
        print('process input image %3.3d/%3.3d' % (i+1, opt.num_test))
        if not opt.sync:
            z_samples = model.get_z_random(opt.n_samples + 1, opt.nz)
        for nn in range(opt.n_samples + 1):
            encode = nn == 0 and not opt.no_encode
            real_A, fake_B, real_B = model.test(z_samples[[nn]], encode=encode)
            if nn == 0:
                images = [real_A, real_B, fake_B]
                #names = ['input' + str(number), 'ground truth' + str(number), 'encoded']
                names = [str(99999 ) , str(9999) , '999']
            else:
                images.append(fake_B)
                #names.append('random_sample%2.2d' % nn)
                names.append(nn)

        img_path = str(number + 101)
        #img_path = 'input_%3.3d' % (number + 101)
        #img_path = ''
        print(img_path)
        save_images(webpage, images, names, img_path, aspect_ratio=opt.aspect_ratio, width=opt.crop_size)

    webpage.save()
 
@app.route('/')  
def upload():  
    return render_template("file_upload_form.html")

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("./results/edges2handbags_2/val/images", filename)

 
@app.route('/success', methods = ['GET', 'POST'])
def success():
    if request.method == 'POST':  
        f = request.files['file']  
        f.save(f.filename)
        image_name = f.filename
        image_number = int(image_name[:3])
        print(image_number)
        source = r'./results/edges2handbags_2/val/images/'
        source_files = os.listdir(source)
        for f1 in source_files:
            os.remove(source + f1)
        test(image_number-101)
        image_names = os.listdir('./results/edges2handbags_2/val/images')
        image_names.sort(reverse=True)
        print(image_names)
        return render_template("gallery.html", image_names=image_names)

      
  
if __name__ == '__main__':  
    app.run(debug = True) 
