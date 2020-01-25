import efficientnet.keras as efn
from pathomix.efficient_net.models_own.eff_net import EffNetFT
from keras.preprocessing.image import ImageDataGenerator


MODEL_PATH = "./model_fine_tuned_with_proper_validation_B0_final"
TILE_FOLDER = '/home/ubuntu/pathomix/data/msi_gi_ffpe_cleaned/CRC_DX/TEST'
INPUT_SIZE = (224, 224)
BATCH_SIZE = 16


def run_inference(model_path, tile_folder, input_size, batch_size):


    # define generator for prediction
    data_provider = ImageDataGenerator(rescale=1./255,
                                      fill_mode='constant',
                                      cval=0
                                      )

    inference_generator = data_provider.flow_from_directory(
            tile_folder,
            target_size=(input_size[0], input_size[1]),
            batch_size=batch_size,
            class_mode='binary'
            )

    # load model
    mymodel = EffNetFT()
    # load model checkpoint
    mymodel.load_model(model_path)

    # perform inference
    predictions = mymodel.model.predict_generator(inference_generator, steps=inference_generator.n//1600, max_queue_size=10, workers=3,
                                    use_multiprocessing=True, verbose=1)

if __name__ == "__main__":
    run_inference(model_path=MODEL_PATH, tile_folder=TILE_FOLDER, input_size=INPUT_SIZE, batch_size=BATCH_SIZE)

