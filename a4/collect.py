"""
collect.py
"""

from TwitterAPI import TwitterAPI
import networkx as nx
import sys
import os
import time
import json
import time
import subprocess


consumer_key = '0uEmGmXVU1NwzvOnPVhPG5pbl'
consumer_secret = 'Bqgap0TYRRwePGLfSSTjdhGA9nVJbmMgeeGtubsjZdSYl8pOGw'
access_token = '3612124212-deDeXB1AXVykupCwJZ8QS8euoT6uDq73lGD8vuX'
access_token_secret = 'GkHrHGyMCpzhC2eHCNQoxfsRusNhYINKspR0HN4D2ODE8'


def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)


def robust_request(twitter, resource, params, max_tries=5):
    for i in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(61 * 15)


def read_screen_names(filename):
    file = open(filename)
    return [line.strip() for line in file]



def get_users(twitter, screen_names):
    params = {'screen_name' : screen_names}
    return robust_request(twitter, 'users/lookup', params)


def get_friends(twitter, screen_name, count = 5000):
    """ 
    Get friends' id
    """
    params = {'screen_name' : screen_name, 'count': count}
    ids = robust_request(twitter, 'friends/ids', params).json()['ids']
    return sorted(list(set(ids)))

def get_followers(twitter, screen_name, count = 5000):
    """
    Get followers' id
    """
    params = {'screen_name': screen_name, 'count': count}
    ids = robust_request(twitter, 'followers/ids', params).json()['ids']
    return sorted(list(set(ids)))


def add_all_friends(twitter, users):
    for u in users:
        u['friends'] = get_friends(twitter, u['screen_name'])


def add_all_followers(twitter, users):
    for u in users:
        u['followers'] = get_followers(twitter, u['screen_name'])





def print_num_friends(users):
    """Print the number of friends per candidate, sorted by candidate name.
    See Log.txt for an example.
    Args:
        users....The list of user dicts.
    Returns:
        Nothing
    """
    sorted(users, key = lambda x: x['screen_name'])
    for u in users:
        print(u['screen_name'], len(u['friends']))



# def get_tweets(twitter, users):
#     tweets = dict()
#     # for u in users:
#     #     tweets[u['screen_name']] = []
#     for u in users:
#         tmptweet = {}
#         for i in range(0, 16):   #iterate through 10 times to get max No. of tweets
#             params = {'screen_name': u['screen_name'], 'include_rts': False, 'count': 200}
#             user_timeline = robust_request(twitter, 'statuses/user_timeline', params)
#             tweet_text = [t['text'] for t in user_timeline]
#             tmptweet = tmptweet | set(tweet_text)
#         tweets[u['screen_name']] = list(tmptweet)
#         # for i in range(0, 16):   #iterate through 10 times to get max No. of tweets
#         #     params = {'screen_name': u['screen_name'], 'include_rts': False, 'count': 200}
#         #     user_timeline = robust_request(twitter, 'statuses/user_timeline', params)
#         #     tweet_text = [t['text'] for t in user_timeline]
#         #     tweets[u['screen_name']].extend(tweet_text)
#     return tweets

def get_tweets(twitter, users):
    tweets = dict()
    for u in users:
        params = {'screen_name': u['screen_name'], 'include_rts': False, 'count': 200}
        user_timeline = robust_request(twitter, 'statuses/user_timeline', params)
        tweet_text = [t['text'] for t in user_timeline]
        tweets[u['screen_name']] = tweet_text
    return tweets



def save_users_to_file(users, path):
    for u in users:
        json.dump(u, open(path + u['screen_name'] + ".txt", 'w'))


def save_tweets_to_file(tweets, path):
    json.dump(tweets, open(path + 'tweets.txt', 'w'))





def main():
    #Install community module
    subprocess.call("./install.sh", shell=True)

    user_save_path = "./new_users/"
    tweets_save_path = "./new_tweets/"

    twitter = get_twitter()
    print('Established Twitter connection.')
    screen_names = read_screen_names('candidates.txt')
    print('Read screen names: %s' % screen_names)
    users = sorted(get_users(twitter, screen_names), key=lambda x: x['screen_name'])
    print('found %d users with screen_names %s' %
          (len(users), str([u['screen_name'] for u in users])))


    print("Now add all friends's ids... ")
    add_all_friends(twitter, users)
    print("Now add all followers' ids...")
    add_all_followers(twitter, users)

    save_users_to_file(users, user_save_path)

    print('Get tweets of each user...')
    tweets = get_tweets(twitter, users)
    save_tweets_to_file(tweets, tweets_save_path)

    # saveresulttofile(tweets, friends2num, followers2num, users)


if __name__ == '__main__':
    main()







