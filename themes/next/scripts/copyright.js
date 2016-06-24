// Add a tail to every post from tail.md
// Great for adding copyright info

var fs = require('fs');

hexo.extend.filter.register('before_post_render', function(data){
    if(data.copyright == false) return data;
    if(data.content.length > 50)
    {
        data.content += '\n<span style="color:red"><em>本文来自<a style="color:red" href="http://sharkdtu.com/">守护之鲨</a>，转载请注明出处：' + data.permalink + '</em></span>';
    }
    return data;
});
