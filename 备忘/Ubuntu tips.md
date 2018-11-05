## Ubuntu

ubuntu支持exfat的方法 

```shell
$ sudo apt-get install exfat-utils
```

ubuntu下锐捷客户端

```shell
$ sudo chmod +x ./rjsupport.sh
$ sudo ./rjsupport.sh -d 1 -u S1807016 -p 036516
$ sudo service network-manager restart
```

## pip源配置：

清华pip源地址：<https://pypi.tuna.tsinghua.edu.cn/simple> 

临时使用：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

[ubuntu更换安装源和pip镜像源](https://blog.csdn.net/wssywh/article/details/79216437)

## Ubuntu下添加应用快捷方式：

[Ubuntu 下添加应用快捷方式](https://blog.csdn.net/yin__ren/article/details/80469499)

## 清华大学开源软件镜像站：

[清华大学开源软件镜像站](https://mirrors.tuna.tsinghua.edu.cn/)

## Git

#### 创建Git仓库


```shell
$ git init
```

#### 添加文件到Git仓库


```shell
$ git add readme.txt
```

#### 提交修改到Git仓库

```shell
$ git commit -m "some info"
```


#### 查看仓库状态


```shell
$ git status
```

#### 查看不同


```shell
$ git diff readme.txt
```

#### 查看提交历史


```shell
$ git log
```


#### 回退版本

```shell
$ git reset --hard HEAD^
```

两个版本就HEAD^^ 多个版本就是HEAD~100这样

或者这样（利用commit的编号）
```shell
$ git reset --hard 1094a
```

#### 查看命令历史

```shell
$ git reflog
```
#### Git的版本回退原理

Git的版本回退速度非常快，因为Git在内部有个指向当前版本的HEAD指针，当你回退版本的时候，Git仅仅是把HEAD从指向要回退的那个版本

#### 撤销修改

```shell
$ git checkout -- readme.txt
```
命令git checkout -- readme.txt意思就是，把readme.txt文件在工作区的修改全部撤销，这里有两种情况：

一种是readme.txt自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；

一种是readme.txt已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。

总之，就是让这个文件回到最近一次git commit或git add时的状态。

```shell
$ git reset HEAD readme.txt
```
git reset命令既可以回退版本，也可以把暂存区的修改回退到工作区。当我们用HEAD时，表示最新的版本

#### 删除文件

```shell
$ git rm test.txt
之后再提交
$ git commit -m "remove test.txt"
如果未提交，并且删错了
$ git checkout -- test.txt
可以从版本库中还原
```

#### 创建分支

```shell
创建并切换
$ git checkout -b dev
相当于
$ git branch dev   这个是创建
$ git checkout dev    这个是切换
```

git checkout命令加上-b参数表示创建并切换

#### 查看当前分支

```shell
$ git branch
```
git branch命令会列出所有分支，当前分支前面会标一个*号。

#### 合并分支

```shell
$ git merge dev
```

#### 删除分支

```shell
$ git branch -d dev
```

