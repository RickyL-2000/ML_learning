如何在本地访问文档

先安装docsify-cli工具:

npm i docsify-cli -g

然后将本项目clone到本地:

git clone https://github.com/ShusenTang/Dive-into-DL-PyTorch.git
cd Dive-into-DL-PyTorch

然后运行一个本地服务器，这样就可以很方便的在http://localhost:3000实时访问文档网页渲染效果。

docsify serve docs

然后会产生一个网址，用浏览器打开该网址即可。