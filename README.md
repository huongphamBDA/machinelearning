# Jenny Huong Pham's Machine Learning Engineering Project 
See [Wiki](https://github.com/huongphamBDA/machinelearning/wiki) for project report.

# Instructions to run the code 
1. Git clone the repository.
```
git clone git@github.com:huongphamBDA/machinelearning.git
```

2. Download dataset `baseball.sql.tar.gz` from [here](https://drive.google.com/file/d/1pXAwQQMJ4TBUasldnm-1HD8PE8WZgi0k/view?usp=sharing).
3. Unpack it.
```
tar -xvzf baseball.sql.tar.gz
```
This should give you `baseball.sql`.

3. Copy `baseball.sql` to `machinelearning` directory
```
cp baseball.sql machinelearning/.
```
4. Change directory to `machinelearning` 
```
cd machinelearning
```

5. Start the containers. The process will take about thirty minutes depending on computers.
```
docker-compose up
```
