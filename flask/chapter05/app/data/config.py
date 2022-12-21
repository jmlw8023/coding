<<<<<<< HEAD


SECRET_KEY = 'some secret words'
DEBUG = True
ITEMS_PER_PAGE = 10



class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'some secret words'
    DATABASE_URI = 'sqlite://:memory:'

class ProductionConfig(Config):
    DATABASE_URI = 'mysql://user@localhost/foo'

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
=======


SECRET_KEY = 'some secret words'
DEBUG = True
ITEMS_PER_PAGE = 10



class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'some secret words'
    DATABASE_URI = 'sqlite://:memory:'

class ProductionConfig(Config):
    DATABASE_URI = 'mysql://user@localhost/foo'

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
>>>>>>> 3efa523ff0c2328ea326da8c6ab7d5be45143578
