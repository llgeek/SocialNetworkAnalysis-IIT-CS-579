"""
sumarize.py
"""

import json
import os
import sys
import glob
from collections import Counter



def main():
    users = [json.load(open(f)) for f in glob.glob(os.path.join("./new_users/", '*.txt'))]
    partition = json.load(open("./partition.txt"))
    tweets = json.load(open("./new_tweets/tweets.txt"))
    sent_tweets = json.load(open("./sentiment_tweets.txt"))
    degree = json.load(open('./degree.txt'))

    with open("summary.txt", 'w', encoding='utf8') as file:
        file.write("User:\t\t#following:\t\t#followers:\n")
        for u in users:
            file.write("%s\t\t%d\t\t%d\n" %(u['screen_name'], len(u['friends']), len(u['followers'])))
        file.write("Total number of users(Remove users with degree < 2): %d\n" %(len(partition)))
        file.write("Degree distribution:\n")
        for i in degree:
            file.write("degree: %s, num: %d\n" %(i, degree[i]))

        file.write("User:\t\t#tweets:\n")
        for t in tweets:
            file.write("%s\t\t%d\n" %(t, len(tweets[t])))
        total_tweet = 0
        for t in tweets:
            total_tweet += len(tweets[t])
        file.write("number of tweets collected: %d\n" %(total_tweet))

        file.write("Number of communities discovered: %d\n" %len(set(partition.values())))
        file.write("Average number of users per community: %d\n" %(len(partition) / len(set(partition.values()))))
        com2num = Counter(partition.values())
        for key in com2num:
            file.write("community %d has %d nodes\n" %(key, com2num[key]))

        file.write("\n\nSentiment analysis results:\n")
        file.write("Top 10 positive tweets for each user\n")
        for u in sent_tweets:
            file.write("\nfor user %s\n:" %u)
            file.write("pos:\t\tneg:\t\ttweets:\n")
            for term in sorted(sent_tweets[u], key = lambda x : (-x['pos'], x['neg']))[:10]:
                file.write("%d\t\t%d\t\t%s" %(term['pos'], term['neg'], term['text']))
                file.write("\n")
        file.write("Top 10 negative tweets for each user\n")
        for u in sent_tweets:
            file.write("\nfor user %s\n:" %u)
            file.write("pos:\t\tneg:\t\ttweets:\n")
            for term in sorted(sent_tweets[u], key = lambda x : (-x['neg'], x['pos']))[:10]:
                file.write("%d\t\t%d\t\t%s" %(term['pos'], term['neg'], term['text']))
                file.write("\n")




if __name__ == "__main__":
    main()