"""
Download comments for a public Facebook post.
"""

import facebook_scraper as fs
import os
import regex as re
import csv

if __name__ == "__main__":
    output_text_f = "/Users/tieuanhnguyen/PycharmProjects/FinalThesis/crawler/text_crawler"
    if not os.path.exists(output_text_f):
        os.mkdir(output_text_f)

    comment_texts = []
    fields = ['free_text', 'label_id']

    POST_ID = "pfbid0MdFXx5b1XdqNseR7aDZvCZEo5Z8sMnLRUpqUpxpyHwuMh4jLyiANJkMDYxzSS4bZl"

    # number of comments to download -- set this to True to download all comments
    MAX_COMMENTS = 800

    # get the post (this gives a generator)
    gen = fs.get_posts(
        post_urls=[POST_ID],
        options={"comments": MAX_COMMENTS, "progress": True}
    )

    # take 1st element of the generator which is the post we requested
    post = next(gen)

    # extract the comments part
    comments = post['comments_full']

    # process comments as you want...
    for comment in comments:
        # e.g. ...print them

        text = comment["comment_text"].replace("\r", " ").replace("\n", " ")
        check = re.search(r"@\S+", text)
        if check is not None:
            end = check.end()
            text = text[end:].strip()
        comment_texts.append([text, 0])
        for reply in comment['replies']:
            text_r = reply["comment_text"].replace("\r", " ").replace("\n", " ")
            check = re.search(r"@\S+", text_r)
            if check is not None:
                end = check.end()
                text_r = text[end:].strip()
            comment_texts.append([text_r, 0])

    with open(os.path.join(output_text_f, "text_crawler_fb_1007_0020.csv"), "w+") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(comment_texts)

    csvfile.close()


