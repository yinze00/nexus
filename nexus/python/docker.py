import requests

def get_docker_tags_count(repo):
    url = f"https://registry.cn-hangzhou.aliyuncs.com/v2/repositories/{repo}/tags"
    response = requests.get(url)
    if response.status_code != 200:
        print(response)
        raise Exception(f"Failed to fetch tags for repository {repo}: {response.status_code}")
    
    data = response.json()
    tags = data['results']
    tag_count = len(tags)

    # Check if there are more pages of results
    while 'next' in data and data['next']:
        response = requests.get(data['next'])
        if response.status_code != 200:
            print(response)
            raise Exception(f"Failed to fetch tags for repository {repo}: {response.status_code}")
        data = response.json()
        tags.extend(data['results'])
        tag_count += len(data['results'])

    return tag_count


if __name__ == "__main__":
    repo = "havenask/ha3_dev"  # 替换为你要查询的仓库名称，例如 "library/ubuntu"
    try:
        count = get_docker_tags_count(repo)
        print(f"The repository '{repo}' has {count} tags.")
    except Exception as e:
        print(e)
