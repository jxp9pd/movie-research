'''Evaluate model performance via visualizations and summary stats'''
from keras.models import Model, load_model
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.image as mpimg
from sklearn.metrics import roc_auc_score

DATA_PATH = '/Users/johnpentakalos/Posters/'
#%%

def predict_df(Y_test, predictions, threshold):
    '''Returns an organized predictions dataset from model predictions.
    Takes a parameter for threshold. 0 returns original values..'''
    genre_df = pd.read_csv(DATA_PATH + 'genres.csv')
    genre_df.set_index('id', inplace=True)
    genre_list = genre_df.columns.values
    actual_df = pd.DataFrame(Y_test, columns=genre_list)
    predictions_df = pd.DataFrame(predictions, columns=genre_list)
    if threshold>0:
        predictions_df[predictions_df < threshold] = 0
        predictions_df[predictions_df > threshold] = 1
    predictions_df.index = actual_df.index
    actual_df.head()
    return predictions_df, actual_df
#%%
def get_precision_recall(predictions_df, actual_df):
    '''Calculate example-based precision metric
    https://stackoverflow.com/questions/9004172/precision-recall-for-multiclass-multilabel-classification'''
    #positives = predictions_df[]
    for genre in predictions_df.columns:
        #Make precision calculations
        positives = predictions_df[genre][predictions_df[genre] == 1]
        if len(positives) == 0:
            print('No positives predicted for: ' + genre)
        else:
            ground_truth = actual_df.loc[positives.index][genre]
            correct = ground_truth.sum()
            print('Precision rate for ', genre, ': ' + str(correct/len(positives)))
        true_labels = actual_df[genre][actual_df[genre] == 1]
        if len(true_labels) == 0:
            print('No true labels exist in the test set for: ' + genre)
        else:
            predicted = predictions_df.loc[true_labels.index][genre]
            caught = predicted.sum()
            print('Recall rate for ', genre, ': ' + str(caught/len(true_labels)))

#%%
def get_auc(y, pred):
    '''Calculate AUC. This means we need predictions_df without thresholding.'''
    for genre in pred.columns:
        genre_y = y[genre]
        genre_pred = pred[genre]
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        
#%%
def loss_curves(history, save_loc, loss_name, acc_name):
    '''Produces loss plots'''
    # summarize history for accuracy
#    plt.plot(history.history['categorical_accuracy'])
#    plt.plot(history.history['val_categorical_accuracy'])
#    plt.title('Model Accuracy')
#    plt.ylabel('accuracy')
#    plt.xlabel('epochs')
#    plt.legend(['train', 'val'], loc='upper left')
#    plt.savefig(save_loc + acc_name + '.png')
#    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(save_loc + loss_name + '.png')
    plt.show()
#%%
def open_img(file_name):
    '''Testing reading an image.'''
    img = mpimg.imread(DATA_PATH + 'poster_imgs/' + file_name + '.jpg')
    plt.imshow(img)
    
#%%