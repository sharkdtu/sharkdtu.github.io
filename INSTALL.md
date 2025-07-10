### Install nodejs

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
nvm install 12.14.0
nvm use 12.14.0
```

### Clone this branch

```bash
$ git clone -b source https://github.com/sharkdtu/sharkdtu.github.io.git
```

### Install node modules

```bash
$ cd sharkdtu.github.io
$ npm install
```

### Generate pages

```bash
$ ./node_modules/hexo/bin/hexo generate
```

### Browse the pages

```bash
$ ./node_modules/hexo/bin/hexo server
```

See http://localhost:4000/ .
