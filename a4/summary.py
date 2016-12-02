import pickle

def main():
    summary_file = open('summary.txt', 'w')
    tweets = pickle.load(open("tweets.pkl", "rb"))
    positive_tweets = pickle.load(open("positive_tweets.pkl", "rb"))
    negative_tweets = pickle.load(open("negative_tweets.pkl", "rb"))
    neutral_tweets = pickle.load(open("neutral_tweets.pkl", "rb"))
    unique_users = set([tweet['user']['screen_name'] for tweet in tweets])

    cluster=[]
    for line in open('cluster_output.txt', 'r'):
        cluster.append(line[:-1])
    num_communities=int(cluster[0])
    avg_per_community=int(cluster[1])/num_communities
    summary_file.write("Number of users collected: " + str(len(unique_users)))
    summary_file.write('\nNumber of messages collected:' + str(len(tweets)))
    summary_file.write('\nNumber of communities detected: ' + str(num_communities))
    summary_file.write('\nAverage number of users per community: ' + str(avg_per_community))
    summary_file.write('\nNumber of instances per class found:\n' + '\tTotal positive tweets instances: ' + str(len(positive_tweets)))
    summary_file.write('\n\tTotal negative tweets instances: ' + str(len(negative_tweets)))
    summary_file.write('\n\tTotal neutral tweets instances: ' + str(len(neutral_tweets)))
    summary_file.write('\nOne example from each class:\n' + '\tPositive sample: ' + str(positive_tweets[0]))
    summary_file.write('\n\tNegative sample: ' + str(negative_tweets[0]))
    summary_file.write('\n\tNeutral sample: ' + str(neutral_tweets[0]))

if __name__ == '__main__':
    main()