# git工具使用笔记

## 1. git安装与用户信息配置

```bash
git config --global user.name pgw
git config --global user.emal 390911793@qq.com
```

git下载地址：[https://git-scm.com/download/win/](https://git-scm.com/download/win)

由于Git是分布式版本控制系统，需要填写用户名和邮箱作为一个标识。**--global**参数表明这台电脑上的所有git仓库都使用相同的配置。用户名的作用是区分不同操作者身份，用户的信息在每一个版本的提交信息中能够看到，以此确认本次提交是谁做的。

## 2. 具体操作

### 创建仓库（repository）

```bash
git init
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619133627763.png" alt="image-20230619133627763" style="zoom:80%;" />

### 提交文件

```bash
# 将文件提交到暂存区
git add README.md 

# 将文件提交到仓库,-m ''未提交时的注释信息
git commit -m ''
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619134634240.png" alt="image-20230619134634240" style="zoom:80%;" />

```bash
# 查看仓库中的文件状态
git status
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619134820360.png" alt="image-20230619134820360" style="zoom:80%;" />

没有暂存的文件未commit到仓库，但是有文件没有添加到仓库中

```bash
# 查看文件的修改内容
git diff RAEDME.md
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619135248835.png" alt="image-20230619135248835" style="zoom:80%;" />

### 回退版本

```bash
# 查看提交记录
git log
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619135946963.png" alt="image-20230619135946963" style="zoom:80%;" />

```bash
# 简化显示
git log --pretty=oneline
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619140158260.png" alt="image-20230619140158260" style="zoom:80%;" />

```bash
# 回退版本
git reset --hard HEAD^
git reset --hard HEAD^^
git reset --hard HEAD~100
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619140454103.png" alt="image-20230619140454103" style="zoom:80%;" />



```bash
# 查看文件内容
cat README.md
```

此时查看提交的日志log，发现只有上上版本的提交记录了。

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619140541693.png" alt="image-20230619140541693" style="zoom:80%;" />

如果要回退到之前的修改版本，则需要修改版本的版本号，使用reflog命令

```bash
git reflog
git reset --hard 44dd381
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619141141776.png" alt="image-20230619141141776" style="zoom:80%;" />

### 撤销修改

1. 修改文件，然后add--->commit

2. 回退到上一个版本 git reset --hard HEAD^

3. checkout命令

   ```bash
   git checkout -- README.md
   ```

   - 修改了后，还没有add到仓库，回到仓库里最新版本的样子
   - 修改了后，已经add到了仓库，又修改了，回退到add到仓库的样子

### 删除文件

```bash
rm README.md
```

只要还没有commit就可以恢复

```bash
git checkout -- README.md
```

## 3. 远程仓库

```bash
# 连接到远程github仓库
git remote add origin https://github.com/I-WI-SH/penguowei.git
```

```bash
# push 
git push -u origin master
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619144612276.png" alt="image-20230619144612276" style="zoom:80%;" />

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619144626937.png" alt="image-20230619144626937" style="zoom:80%;" />

```bash
# clone远程仓库
git clone https://github.com/I-WI-SH/penguowei.git
```

## 4. 创建与合并分支

```bash
# 创建并切换分支
git brach dev
git checkout dev
git checkout -b dev

# 查看当前仓库有多少分支
git branch
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619150648002.png" alt="image-20230619150648002" style="zoom:80%;" />

在dev分支做出的修改commit到仓库后，切换到master分支后，master分支下的内容不会变；如果要使master分支下的内容和dev分支下的内容一样，则要将分支进行合并

```bash
# 合并分支
git merge dev
```

```bash
# 删除分支
git branch -d dev
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619152309277.png" alt="image-20230619152309277" style="zoom:80%;" />

当在不同分支下，对文件中的同一位置做了修改后，在合并分支时会发生冲突

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619153242992.png" alt="image-20230619153242992" style="zoom:80%;" />

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619153325368.png" alt="image-20230619153325368" style="zoom:80%;" />

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619153435363.png" alt="image-20230619153435363" style="zoom:80%;" />

确认修改冲突内容后，重新commit

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619154052363.png" alt="image-20230619154052363" style="zoom:80%;" />

```bash
git merge -no-off -m 'commit_info' dev
```

在一般merge模式下，当删除分支后，分支信息将全部消失；在-no-off模式下进行merge，删除分支后，分支的版本信息依旧存在

## 5. 临时分支

当有一些分支的内容暂时不想提交到代码仓库中，而一部分需要提交进行开发维护使用stash命令

```bash
git stash
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619155502899.png" alt="image-20230619155502899" style="zoom:80%;" />

工作区保存的位置为最近一次commit

```bash
# 恢复工作区
git stash pop

git stash list
git stash drop
```

## 6. 多人协作

```bash
# 查看远程仓库信息
git remote
git remote -v
```

<img src="C:\Users\39091\Desktop\penguowei\week1\git工具使用笔记\图片\image-20230619155838171.png" alt="image-20230619155838171" style="zoom:80%;" />

```bash
# 推
git push origin [分支]

# 抓
git clone [地址]
```

```bash
# 解决远程dev分支和本地dev分支冲突问题
git branch --set-upstream dev origin/dev
git pull
```

