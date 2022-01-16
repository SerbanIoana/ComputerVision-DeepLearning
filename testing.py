from data_loading import load_data_set

if __name__ == "__main__":
    # load test data
    test_descriptions, test_features = load_data_set('test', 'dataset\Flickr8k_text\Flickr_8k.testImages.txt')
    print(f'Descriptions: test={len(test_descriptions)}')
    print(f'Photos: test={len(test_features)}')
