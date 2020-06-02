""" Creating a user class """

class User():
    def __init__(self, username, password, bio, posts=[]):
        self.username = username
        self.password = password
        self.bio = bio
        self.posts = posts
    def new_post(self):
        post = "New Status"
        posts.append(post)
