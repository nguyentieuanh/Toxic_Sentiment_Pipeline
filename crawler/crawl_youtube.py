from itertools import islice
from youtube_comment_downloader import *
import os
import re
import csv

if __name__ == "__main__":
    output_text_f = "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler"
    if not os.path.exists(output_text_f):
        os.mkdir(output_text_f)

    downloader = YoutubeCommentDownloader()
    comments = downloader.get_comments_from_url(
        'https://www.youtube.com/watch?v=0OWNZ8TYbkg',
        sort_by=SORT_BY_POPULAR)
    fields = ['free_text', 'label_id']
    comment_texts = []
    for comment in comments:
        text = comment['text']
        check = re.search(r"@\S+", comment['text'])
        if check is not None:
            end = check.end()
            text = text[end:].strip()
        comment_texts.append([text, 0])

    with open(os.path.join(output_text_f, "text_crawler_2906.csv"), "w+") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(comment_texts)

    csvfile.close()
