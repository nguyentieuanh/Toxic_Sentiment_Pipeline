import pandas as pd


# train_data = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/train_toxic_multilabel.csv")
#
# gen_text_label0 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_toxic_label0.csv")
# gen_text_label1 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_suicide_label1.csv")
# gen_text_label2 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_harrasment_label2.csv")
# gen_text_label3 = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/valid/valid_gen_gender_label3.csv")
#
#
# train_new_data = pd.concat([train_data, gen_text_label0, gen_text_label1, gen_text_label2, gen_text_label3], ignore_index=True)
# train_new_data.to_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/valid_multilabel_toxic.csv")
#
def get_text_label(samples_clean):
    df = pd.read_csv("/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/train.csv")
    df = df[df['free_text'].notnull()]
    df_hate = df[df['label_id'] == 1]
    df_random_clean = df[df['label_id'] == 0].sample(n=samples_clean, random_state=123)
    df_new = df_random_clean
    # df_new = pd.concat([df_random_clean, df_hate])
    df_new_shuffle = df_new.sample(frac=1, random_state=234).reset_index(drop=True)
    df_new_shuffle.to_csv(
        "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/data_train_update.csv")


def concat_file(file_path_list):
    df_list = []
    for pth in file_path_list:
        df_list.append(pd.read_csv(pth))
    train_new_data = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
    train_new_data = train_new_data.drop_duplicates()
    # train_new_data["label_id"] = train_new_data["label_id"].replace([2, 3, 4], 1)
    train_new_data = train_new_data.sample(frac=1, random_state=234).reset_index(drop=True)
    print(train_new_data.replace(2, 1)["label_id"].value_counts())
    train_new_data.to_csv(
        "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/test_text_crawler/test_new_ver3.csv")


def concat_file_toxic(file_path_list):
    df_list = []
    for pth in file_path_list:
        df_list.append(pd.read_csv(pth))
    train_new_data = pd.concat(df_list, ignore_index=True)
    train_new_data = train_new_data.drop_duplicates()
    train_new_data["label_id"] = train_new_data["label_id"].replace([2, 3, 4], 1)
    train_new_data = train_new_data.sample(frac=1, random_state=234).reset_index(drop=True)
    train_new_data = train_new_data[train_new_data["label_id"] == 1]
    print(train_new_data.replace(2, 1)["label_id"].value_counts())
    train_new_data.to_csv(
        "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/test_text_crawler/test_new_toxic.csv")


file_list = ["/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/data_train_update.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1640.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1650.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1704.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/train/train_gen_harrassment_label2.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/train/train_gen_tu_tu_label1.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/train/train_gen_gender_label3.csv",
             "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/text_gen/train/train_gen_toxic_label4.csv"]

file_list_toxic = [
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_tu_tu_0907_1752.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_tu_tu_0907_1856.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2008.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2028.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2041.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2114.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2125.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2143.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2205.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_youtube_0907_2248.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_tu_tu_0907_1910.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_gender_0907_1910.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_0907_2337.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_1007_0003.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_youtube_0907_2248.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_youtube_0907_2305.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_fb_1007_0020.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_2906_1745.csv",
    # "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_2906.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1627.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1640.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1650.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0507_1704.csv",
    "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler/text_crawler_0907_2008.csv"
]

file_list_concat_test = ["/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/test_text_crawler/test_new_toxic.csv",
                         "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/test_text_crawler/test_new.csv",
                         "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/synonyms_upsampling/vihsd/test.csv"]

if __name__ == "__main__":
    # get_text_label(15000)
    concat_file(file_list_concat_test)
