// Add a tail to every post from tail.md
// Great for adding copyright info

var fs = require('fs');

hexo.extend.filter.register('before_post_render', function(data){
    if(data.copyright == false) return data;
    if(data.content.length > 50) 
    {
        data.content += '\n*转载请注明出处, 本文永久链接：' + data.permalink + '*';
    }
    return data;
});
