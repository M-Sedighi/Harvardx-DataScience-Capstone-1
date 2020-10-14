#########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rmarkdown)) install.packages("rmarkdown", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(rmarkdown)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#########################################################
# Preprocessing the data and preparing for the algorithm
##########################################################

# adding a column to both training and validation set containing the date of the review in year
org <- as.Date('1970-1-1') # define the date origin for conversion based on the Movielens database documentation

set <- edx %>% mutate(date = as.Date(timestamp/86400,org)) %>% mutate(date=format(date, format="%Y"))
holdout <- validation %>% mutate(date = as.Date(timestamp/86400,org)) %>% mutate(date=format(date, format="%Y"))

set.seed(2, sample.kind="Rounding") 
test_index <- createDataPartition(y = set$rating, times = 1, p = 0.1, list = FALSE)
train_set <- set[-test_index,]
temp <- set[test_index,]

# Make sure userId and movieId in validation set are also in edx set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

# function for calculating RSME              
RMSE <- function(true_ratings, predicted_ratings){sqrt(mean((true_ratings - predicted_ratings)^2))}

#########################################################
# running the algorithm as a function for different values of lambda
##########################################################


# define the function for calculating the RMSE with lambda as input

model <- function(l){
  mu <- mean(train_set$rating) # calculating the mean of all of ratings to use as a baseline for predictions
  b_i <- train_set %>% group_by(movieId) %>% summarize(b_i = sum(rating - mu)/(n()+l)) # calculating the regularized movie bias
  ts <- train_set %>% left_join(b_i, by="movieId") # adding movie bias column to the training data
  b_u <- ts %>% group_by(userId) %>% summarize(b_u = sum(rating - b_i - mu)/(n()+l)) # calculating the regularized user bias
  ts <- ts %>% left_join(b_u, by = "userId") # adding user bias column to the training data
  b_g <- ts %>% group_by(genres) %>% summarize(b_g =sum(rating - b_i - b_u - mu)/(n()+l)) # calculating the regularized genres bias
  ts <- ts %>% left_join(b_g, by = "genres") # adding genres bias column to the training data
  # calculating the review year bias  per movie 
  b_d_i <- ts %>% group_by(movieId,date) %>% summarize(b_d_i =sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  
  prediction <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_d_i, by = c("movieId"="movieId","date"="date"))
  
  prediction$b_d_i[is.na(prediction$b_d_i)] <- 0
  predicted_ratings <- prediction %>% mutate(pred = mu + b_i + b_u + b_g + b_d_i) %>% .$pred
  return(RMSE(predicted_ratings, test_set$rating))
}

# defining the lambdas for the first stage of analysis
lambdas <- seq(1, 10, 1)

# applying the function on training and test set with first stage lambdas
rmses <- sapply(lambdas, model)

# finding the best lambda value from the first stage
result <- data.frame(Lambda=lambdas, RMSE=rmses)
print(result)

lambda <- lambdas[which.min(rmses)]
lambda
min(rmses)
plot(result)

# defining the lambdas for the second stage of analysis
lambdas <- seq(lambda-0.9, lambda+0.9, 0.1)

# applying the function on training and test set with second stage lambdas
rmses <- sapply(lambdas, model)

# finding the best lambda value from the second stage
result <- data.frame(Lambda=lambdas, RMSE=rmses)
print(result)
plot(result)
lambda <- lambdas[which.min(rmses)]
lambda
min(rmses)

#########################################################
# final validation and calculation of RMSE for the holdout set
##########################################################

# calculating the RMSE for the final validation set using the best lambda found in second stage

train_set <- set
test_set <- holdout

final_RMSE <- model(lambda)
print(final_RMSE)


