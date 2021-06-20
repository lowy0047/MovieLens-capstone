##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(dplyr)
library(knitr)
library(stringr)
library(lubridate)
library(tinytex)

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

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#Q1
nrow(edx)
ncol(edx)

#Q2
edx %>% filter(rating==0) %>% nrow()
edx %>% filter(rating==3) %>% nrow()

#Q3
nrow(distinct(data.frame(edx$movieId)))

#Q4
nrow(distinct(data.frame(edx$userId)))

#Q5
edx %>% filter(str_detect(genres, "Drama")) %>% nrow()
edx %>% filter(str_detect(genres, "Comedy")) %>% nrow()
edx %>% filter(str_detect(genres, "Thriller")) %>% nrow()
edx %>% filter(str_detect(genres, "Romance")) %>% nrow()

#Q6
edx %>% group_by(title) %>% summarize(n=n()) %>% arrange(desc(n))

#Q7
edx %>% group_by(rating) %>% summarize(n=n()) %>% arrange(desc(n)) %>% slice(1:5)

#Q8
table(edx$rating)

# Extract release year from title
edx <- edx %>%
  mutate(release_year = as.numeric(str_sub(title, -5, -2)))
validation <- validation %>% 
  mutate(release_year = as.numeric(str_sub(title, -5, -2)))

# Extract rating year from timestamp then delete timestamp
edx <- edx %>% mutate(rating_year = year(as_datetime(timestamp)))
validation <- validation %>% mutate(rating_year = year(as_datetime(timestamp)))
edx <- edx %>% select (-c(timestamp))
validation <- validation %>% select (-c(timestamp))

# Plot movie release year against rating count
edx %>% ggplot(aes(release_year)) +
  geom_histogram(bins=30, fill="black", col="grey") +
  labs(title = "Movie release year against rating count", x = "Release year", 
       y = "Number of ratings")

# Plot movie release year vs average ratings
edx %>% group_by(release_year) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(release_year, avg_rating)) +
  geom_point() +
  labs(title = "Movie average rating by release year", x = "Release year", 
       y = "Average rating")

# Plot movie rating year against rating count
edx %>% ggplot(aes(rating_year)) +
  geom_histogram(bins=30, fill="black", col="grey") +
  labs(title = "Movie rating year against rating count", x = "Rating year", 
       y = "Number of ratings")

# Plot movie rating year vs average ratings
edx %>% group_by(rating_year) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(rating_year, avg_rating)) +
  geom_line() +
  labs(title = "Movie average rating by rating year", x = "Rating year", 
       y = "Average rating")

# Unpivot data to obtain one row for each genre
edx_unpivot_genres <- edx %>% 
  mutate(genre=fct_explicit_na(genres, na_level = "(None)")) %>%
  separate_rows(genre,sep = "\\|")

# Plot genre against rating count
edx_unpivot_genres %>%
  group_by(genre) %>%
  summarize(n=n()) %>%
  ggplot(aes(x=genre, y=n)) +
  geom_bar(stat='identity') +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Genre against rating count", x = "Genres", y = "Number of ratings")

# Plot movie genre vs average ratings
edx_unpivot_genres %>% group_by(genre) %>%
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(genre, avg_rating)) +
  geom_point() +
  theme(axis.text.x = element_text(angle = 90)) +
  labs(title = "Genre average ratings", x = "Genres", y = "Average rating")

# Plot movie rating count distribution frequency
edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=30, fill="black", col="grey") +
  scale_x_log10() +
  labs(title = "Movie rating count distribution frequency", x = "Movie rating count", 
       y = "Number of movies")

# Plot user rating count distribution frequency
edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins=30, fill="black", col="grey") +
  scale_x_log10() +
  labs(title = "User rating count distribution frequency", x = "User rating count", 
       y = "Number of movies")

# First model
mu_hat <- mean(edx$rating)
model_1_rmse <- RMSE(validation$rating, mu_hat)
rmse_table <- tibble(Method = "Baseline average model", RMSE = model_1_rmse)
knitr::kable((rmse_table), "simple")

# Modeling release year effects
mu <- mean(edx$rating)
release_year_avg <- edx %>%
  group_by(release_year) %>%
  summarize(b_i = mean(rating - mu))
predicted_ratings <- mu + validation %>%
  left_join(release_year_avg, by='release_year') %>%
  pull(b_i)
model_2_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_table <- rbind(rmse_table, tibble(Method = "Release year model", 
                                        RMSE = model_2_rmse))
knitr::kable((rmse_table), "simple")

# Modeling rating year and release year effects
rating_year_avg <- edx %>%
  left_join(release_year_avg, by='release_year') %>%
  group_by(rating_year) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings <- validation %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
model_3_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_table <- rbind(rmse_table, tibble(Method = "Rating and release year model", 
                                       RMSE = model_3_rmse))
knitr::kable((rmse_table), "simple")

# Modeling genre, rating year and release year effects
genres_avg <- edx %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  group_by(genres) %>%
  summarize(b_o = mean(rating - mu - b_i - b_u))
predicted_ratings <- validation %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  left_join(genres_avg, by='genres') %>%
  mutate(pred = mu + b_i + b_u + b_o) %>%
  pull(pred)
model_4_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_table <- rbind(rmse_table, tibble(Method = "Genre, rating and release year model", 
                                       RMSE = model_4_rmse))
knitr::kable((rmse_table), "simple")

# Modeling movie, genre, rating year and release year effects
movie_rating_count_avg <- edx %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  left_join(genres_avg, by='genres') %>%
  group_by(movieId) %>%
  summarize(b_a = mean(rating - mu - b_i - b_u - b_o))
predicted_ratings <- validation %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  left_join(genres_avg, by='genres') %>%
  left_join(movie_rating_count_avg, by='movieId') %>%
  mutate(pred = mu + b_i + b_u + b_o + b_a) %>%
  pull(pred)
model_5_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_table <- rbind(rmse_table, 
                    tibble(Method = "Movie, genre, rating and release year model", 
                           RMSE = model_5_rmse))
knitr::kable((rmse_table), "simple")

# Modeling movie, user, genre, rating year and release year effects
user_rating_count_avg <- edx %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  left_join(genres_avg, by='genres') %>%
  left_join(movie_rating_count_avg, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_e = mean(rating - mu - b_i - b_u - b_o - b_a))
predicted_ratings <- validation %>%
  left_join(release_year_avg, by='release_year') %>%
  left_join(rating_year_avg, by='rating_year') %>%
  left_join(genres_avg, by='genres') %>%
  left_join(movie_rating_count_avg, by='movieId') %>%
  left_join(user_rating_count_avg, by='userId') %>%
  mutate(pred = mu + b_i + b_u + b_o + b_a + b_e) %>%
  pull(pred)
model_6_rmse <- RMSE(predicted_ratings, validation$rating)
rmse_table <- rbind(rmse_table, 
                    tibble(Method = "User, movie, genre, rating and release year model", 
                           RMSE = model_6_rmse))
knitr::kable((rmse_table), "simple")

# Regularization of final model
lambda <- seq(0, 10, 0.25)

rmses <- sapply(lambda, function(l){
  
  mu <- mean(edx$rating)
  
  release_year_avg <- edx %>%
    group_by(release_year) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  rating_year_avg <- edx %>%
    left_join(release_year_avg, by='release_year') %>%
    group_by(rating_year) %>%
    summarize(b_u = sum(rating - mu - b_i)/(n()+l))
  
  genres_avg <- edx %>%
    left_join(release_year_avg, by='release_year') %>%
    left_join(rating_year_avg, by='rating_year') %>%
    group_by(genres) %>%
    summarize(b_o = sum(rating - mu - b_i - b_u)/(n()+l))
    
  movie_rating_count_avg <- edx %>%
    left_join(release_year_avg, by='release_year') %>%
    left_join(rating_year_avg, by='rating_year') %>%
    left_join(genres_avg, by='genres') %>%
    group_by(movieId) %>%
    summarize(b_a = sum(rating - mu - b_i - b_u - b_o)/(n()+l))
  
  user_rating_count_avg <- edx %>%
    left_join(release_year_avg, by='release_year') %>%
    left_join(rating_year_avg, by='rating_year') %>%
    left_join(genres_avg, by='genres') %>%
    left_join(movie_rating_count_avg, by='movieId') %>%
    group_by(userId) %>%
    summarize(b_e = sum(rating - mu - b_i - b_u - b_o - b_a)/(n()+l))
  
  predicted_ratings <- validation %>%
    left_join(release_year_avg, by='release_year') %>%
    left_join(rating_year_avg, by='rating_year') %>%
    left_join(genres_avg, by='genres') %>%
    left_join(movie_rating_count_avg, by='movieId') %>%
    left_join(user_rating_count_avg, by='userId') %>%
    mutate(pred = mu + b_i + b_u + b_o + b_a + b_e) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

rmse_table <- rbind(rmse_table, 
                    tibble(Method = "Regularized final model",
                           RMSE = min(rmses)))

knitr::kable((rmse_table), "simple")
