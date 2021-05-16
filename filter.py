from tqdm import tqdm
import pandas as pd


def filter_data(filename, subreddit='SuicideWatch', timeperiod=604800):

    posts_df = pd.read_csv(
        filename)
    posts_df.head()

    filter_range = posts_df[posts_df['subreddit'] ==
                            'SuicideWatch'][['user_id', 'timestamp', 'subreddit']]

    filter_range['min_tstamp'] = filter_range['timestamp'] - \
        timeperiod
    filter_range['max_tstamp'] = filter_range['timestamp'] + \
        timeperiod

    user_filter = {}
    for idx, r in filter_range.iterrows():
        if not user_filter.get(r.user_id):
            user_filter[r.user_id] = []
        user_filter[r.user_id].append([r.min_tstamp, r.max_tstamp])

    temp = []
    for uid, time_ranges in tqdm(user_filter.items()):
        for rnge in time_ranges:
            temp.append(posts_df[(posts_df['user_id'] == uid) & (
                posts_df['timestamp'].between(rnge[0], rnge[1]))])

    temp_df = pd.concat(temp)

    write_file = filename[:-4] + '_24hours.csv'
    temp_df.to_csv(write_file)


def filter_sw_only(filename):
    posts_df = pd.read_csv(
        filename)
    posts_df.head()

    temp_df = posts_df[posts_df['subreddit'] == 'SuicideWatch']

    write_file = filename[:-4] + '_swonly.csv'
    temp_df.to_csv(write_file)


if __name__ == '__main__':
    '''
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv",
                "SuicideWatch", 1)
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv",
                "SuicideWatch", 1)
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/expert_posts.csv",
                "SuicideWatch", 1)

    filter_sw_only(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv")
    filter_sw_only(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv")
    filter_sw_only(
        "CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/expert_posts.csv")
    '''
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/test/shared_task_posts_test.csv",
                "SuicideWatch", 86400)
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/crowd/train/shared_task_posts.csv",
                "SuicideWatch", 86400)
    filter_data("CL class project materials/umd_reddit_suicidewatch_dataset_v2/expert/expert_posts.csv",
                "SuicideWatch", 86400)
