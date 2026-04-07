import pandas as pd
import os

def join_annotations(base_path):
    annotation_training = pd.DataFrame(pd.read_pickle(os.path.join(base_path, 'annotation_training.pkl')))
    annotation_validation = pd.DataFrame(pd.read_pickle(os.path.join(base_path, 'annotation_validation.pkl')))
    annotation_test = pd.DataFrame(pd.read_pickle(os.path.join(base_path, 'annotation_test.pkl')))

    annotation_training['phase'] = 'train'
    annotation_validation['phase'] = 'validation'
    annotation_test['phase'] = 'test'

    annotation = pd.concat([annotation_training, annotation_validation, annotation_test])

    annotation = annotation[['phase', 'extraversion', 'neuroticism', 'agreeableness', 'conscientiousness', 'openness']]
    annotation.reset_index(inplace=True)
    annotation.rename(columns={'index': 'video_name'}, inplace=True)
    annotation.to_csv('annotation.csv', index=False)


if __name__ == '__main__':
    # Arahkan ke folder Dataset_TA di Google Drive kamu
    join_annotations('D:\\0. Kuliah\\1. Tugas Akhir\\ta-luszark005\\dataset_processing')