import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name
# 현재 리전값을 받아옴

downloaded_data_bucket = f"jumpstart-cache-prod-{region}"
downloaded_data_prefix = "1p-notebooks-datasets/mnist"
# MNIST 데이터셋을 받아올 경로 설정

role = get_execution_role()
bucket = 'sagemaker-sumin-test'
# 아까 만든 IAM role과 S3 Bucket 가져오기


%%time
# python이 아닌 jupyter 노트북의 기능
# 해당 cell의 performance 체크
import pickle, gzip, numpy, urllib.request, json
# pickle : python식 데이터 압축 포맷
# numpy : 수치 계산을 위한 python package

s3 = boto3.client("s3")
s3.download_file(downloaded_data_bucket, f"{downloaded_data_prefix}/mnist.pkl.gz", "mnist.pkl.gz")
with gzip.open("mnist.pkl.gz", "rb") as f:
    train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
# Load DataSet


%matplotlib inline
# matplotlib로 그리는 그림이 jupyter 노트북에 바로 보여줄 수 있도록 설정
import matplotlib.pyplot as plt
# 도표나 그림을 그릴 수 있게 해주는 라이브러리 import

plt.rcParams["figure.figsize"] = (2, 10)

def show_digit(img, caption="", subplot=None):
    if subplot is None:
        _, (subplot) = plt.subplots(1, 1)
    imgr = img.reshape((28, 28))
    subplot.axis("off")
    subplot.imshow(imgr, cmap="gray")
    plt.title(caption)
# 데이터셋의 데이터를 확인하는 함수 작성

show_digit(train_set[0][30], f"This is a {train_set[1][30]}")
# show_digit 함수 테스트 : train_set 30번째 데이터의 그림[0]과 데이터 이름[1] 확인


from sagemaker import KMeans

data_location = 's3://{}/kmeans_highlevel_example/data' .format(bucket)
output_location = 's3://{}/kmeans_example/output' .format(bucket)
# 학습 위해 학습 알고리즘 및 데이터 경로 설정

print('training data will be uploaded to : {}' .format(data_location))
print('training artifacts will be uploaded to : {}' .format(output_location))
# 설정한 경로 출력

kmeans = KMeans(role=role,
               train_instance_count=2,
               train_instance_type='ml.c4.8xlarge',
               output_path=output_location,
               k=10,
               data_location=data_location)
# 학습하는 방법 세팅 : ml.c4.8xlarge 인스턴스 2대로 학습할 것
# k값에 대한 설명은 위에 썼었다


%%time

kmeans.fit(kmeans.record_set(train_set[0]))
# 머신러닝 학습시키기


%%time

kmeans_predictor = kmeans.deploy(initial_instance_count=1,
                                instance_type='ml.m4.xlarge')
# 만든 모델 배포



# valid_set의 30번째 데이터로 sample 테스트

result=kmeans_predictor.predict(valid_set[0][30:31])
print(result)
# predict 함수 : 새로운 이미지가 어떤 cluster에 속했는지 예측 결과를 알려줌


%%time

result=kmeans_predictor.predict(valid_set[0][0:100])
# valid_set에 있는 0부터 99번까지의 cluster 예측하기

clusters=[r.label['closest_cluster'].float32_tensor.values[0] for r in result]
# 예측 결과에 대한 cluster 정보 수집



for cluster in range(10):
    print('\n\n\nCluster {}:'.format(int(cluster)))
    digits = [ img for l, img in zip(clusters, valid_set[0]) if int(l) == cluster ]
    height = ((len(digits)-1)//5)+1
    width = 5
    plt.rcParams["figure.figsize"] = (width,height)
    _, subplots = plt.subplots(height, width)
    subplots = numpy.ndarray.flatten(subplots)
    for subplot, image in zip(subplots, digits):
        show_digit(image, subplot=subplot)
    for subplot in subplots[len(digits):]:
        subplot.axis('off')

    plt.show()
# 분류된 클러스터별로 데이터 출력

