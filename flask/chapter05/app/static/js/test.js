<<<<<<< HEAD


function choosefiles() {
    console.log("选择文件");
    // 获取选择的文件files 这里没有添加是不是图片文件的判断 略
    var files =  document.getElementById('fileImage').files;
    // 获取表格的tbody 后面可以通过它的insertRow等方法实现对表格行列的增加
    var tb = document.getElementById("imgViewtbody");
    var length=files.length;   // 选择的文件数量  
    console.log("选择的文件数量是",length);

    // 判断表格中已经存在的图片 后面继续选择索引 
    // 表格中的图片是按照两列摆放的如下。因此可以根据它的索引i确定这张图应该放在哪一行哪一列
    // 图片0 | 图片1
    // 图片2 | 图片3 

    var rowsNum = tb.rows.length; // 已经表格存在的行数
    var totalitemNum = rowsNum *2; // 总的图片数量 这个值是假设每一行都是两张图的情况下  但实际上最后一行可能值放了第一列的图片
    if(rowsNum >0){
       // 判断最后一行是不是两张图  注意tb的索引是从0开始 rowsNum表示长度，所以最后一行的索引是rowsNum-1
        if(tb.rows[rowsNum-1].cells.length == 1){
            totalitemNum = totalitemNum -1; // 最后一行只有一张图 显然总的数量应该减1
        }
    }

    $.each(files, function(i, file) {  // 这里用each去遍历files 不用for(var i=0;i< files.length;i++)这种方法，因为循环里面用到了一个re.onload方法，它是异步的，还么有看到js的异步和同步处理，只是在网上看到这个解释。用for循环遍历的话，每次取出的值是不正常的，图片只能读取到最后一个
        console.log("遍历",i, file);
        // files的索引是0开始 但是表格中已经右totalitemNum个图片 因此索引加上
        i = i + totalitemNum;
        // imghtml参考的是上面提到的博客的代码  
        var imghtml = '<div id="uploadList_'+ i +'" class="upload_append_list"><p><strong>' + file.name.substring(0,10) + '</strong>&nbsp'+ 
        '<a href="javascript:" class="upload_delete" title="删除" data-index="'+ i +'">删除</a><br />' +
        '<img name = "docfiless" id="uploadImage_' + i +'" class="upload_image" height="256" width="450"/></p>'+
        '<span id="uploadProgress_' + i + '" class="upload_progress">'+'<p>Analyse</p><span></span>'+'<span></span>'+'<span></span>'+'</span>' +
    '</div>'+'<div id="result" class="upload_result"></div>';
        // 根据图片索引的特殊 可以看到偶数都是在第一列 奇数序号图片都是在第二列 如果是偶数 需要给表格新建一行存放图片
        if(i%2==0){
            console.log("新建一行",i);
            let newRow = tb.insertRow(i/2); // 序号i的图片对应的是i/2行图片
            let newcell = newRow.insertCell(0); // 然后在新建的这一行，新建第一列
            newcell.innerHTML = imghtml;  // 那么该行该列的内容就是上面的imghtml
            newcell.style.align = "center";
        }
        else{
            console.log("在原始行添加",i); // 奇数直接在它对应的那一行新加一列即可
            let y = tb.rows[Math.floor(i/2)].insertCell(1);
            y.innerHTML = imghtml;
            y.style.align="center";
        }

        // 上面完成了，但是可以看到imghtml中的图片 还没有定义它的src属性，所以还需要设置src属性
        var imgid = document.getElementById("uploadImage_"+i);  // 取出对应的图片标签

        var re = new FileReader();
        re.onload = function(re) {
            console.log("imgid获取序号",i);  // 用for循环遍历files的话 可以看到这里的i始终将是最后一个 就出现问题
            if(imgid){ 
                imgid.src = re.target.result;  // re.target.result就是读出的文件base64编码字符串
            }
        }
        re.readAsDataURL(file);

        // imghtml中也添加了一个删除操作  因此对它绑定一个删除函数 函数参数是i 表示要删除的图片的序号
        $("#uploadList_"+i+ "> p > a").click(function(){
            deleteItem(i); // 传入的参数是索引 从0开始
        });
    });
}


// 获取当前表格中的图片数量
function getCurItemNums(){
    var tb = document.getElementById("imgViewtbody");
    var rowsNum = tb.rows.length; // 已经表格存在的行数
    var totalitemNum = rowsNum *2; // 总的图片数量 这个值是假设每一行都是两张图的情况下
    if(rowsNum >0){
        if(tb.rows[rowsNum-1].cells.length == 1){
            totalitemNum = totalitemNum -1; // 最后一行只有一张图
        }
    }
    return totalitemNum;// 当前的图像数量
}

// 删除索引为index的图片   分析：删除索引index的图片后，那么索引在它后面的图 应该往前移动一个格子 即它后面的图 ，原先在第一列，应该移动到上一行的第二列  原先在第二列，应该移动到同一行的第一列
// 同时 表格中的原先最后一个图片的位置 应该空出来了
function deleteItem(index){
    console.log("删除第",index,"个");  // 序号从0开始  index对应的是第

    var tb = document.getElementById("imgViewtbody");
    var curnums = getCurItemNums(); // 当前的图像数量

    if(curnums == index+1){ // 如果删除的本身就是最后一个元素 直接删除即可 不会影响到表格中的其他元素
        var row = Math.floor(index/2) ;// 第row行
        var cell = index%2; // 第cell列   索引index图片 对应的行和列是row和cell
        tb.rows[row].deleteCell(cell); // 直接删除它就可以了  // 如果这一行本身只有一个元素的话 执行后 会变成一个空的<tr></tr>， 下一次对其他行列的图片删除的时候 getCurItemNums函数的结果返回就不对
        if(tb.rows[row].cells.length == 0){
            tb.deleteRow(row); // 当前列没有元素的话，再把当前行删除了。这样才是完整的删除 不会有空的<tr>标签在
        }
    }
    else{
        // 如果不是最后一个元素 那么从index开始 ，依次用index+1的图片 替换掉当前index位置的图片即可
        for(var i=index;i<curnums-1;i++){
            // 让索引i的图像变为 i+1的图像  遍历到curnums-1即可 因为最后一张图后面没有了，直接删除最后一张即可
            console.log("修改图像",i)
            let row = Math.floor(i/2) ;// 第row行
            let cell = i%2; // 第cell列
    
            var imgid = $("#uploadImage_"+i)[0]; //取出索引i这个图片的标签 修改它的src 
            imgid.src = $("#uploadImage_"+(i+1)).attr("src"); // 新的src是它下一个的src

            // 还要修改每个表格中显示的文件名 #uploadList_2 > p > strong
            var filename = $("#uploadList_"+i+ "> p > strong")[0];
            filename.innerHTML = $("#uploadList_"+(i+1)+ "> p > strong").html();
        }

        // 最后删除最后一个元素  同样需要判断最后一行是不是空的tr然后删除行
        var row = Math.floor((curnums-1)/2) ;// 第row行
        var cell = (curnums-1)%2; // 第cell列
        tb.rows[row].deleteCell(cell); // 直接删除它就可以了
        if(tb.rows[row].cells.length == 0){
            tb.deleteRow(row); 
        }
    }
}
=======


function choosefiles() {
    console.log("选择文件");
    // 获取选择的文件files 这里没有添加是不是图片文件的判断 略
    var files =  document.getElementById('fileImage').files;
    // 获取表格的tbody 后面可以通过它的insertRow等方法实现对表格行列的增加
    var tb = document.getElementById("imgViewtbody");
    var length=files.length;   // 选择的文件数量  
    console.log("选择的文件数量是",length);

    // 判断表格中已经存在的图片 后面继续选择索引 
    // 表格中的图片是按照两列摆放的如下。因此可以根据它的索引i确定这张图应该放在哪一行哪一列
    // 图片0 | 图片1
    // 图片2 | 图片3 

    var rowsNum = tb.rows.length; // 已经表格存在的行数
    var totalitemNum = rowsNum *2; // 总的图片数量 这个值是假设每一行都是两张图的情况下  但实际上最后一行可能值放了第一列的图片
    if(rowsNum >0){
       // 判断最后一行是不是两张图  注意tb的索引是从0开始 rowsNum表示长度，所以最后一行的索引是rowsNum-1
        if(tb.rows[rowsNum-1].cells.length == 1){
            totalitemNum = totalitemNum -1; // 最后一行只有一张图 显然总的数量应该减1
        }
    }

    $.each(files, function(i, file) {  // 这里用each去遍历files 不用for(var i=0;i< files.length;i++)这种方法，因为循环里面用到了一个re.onload方法，它是异步的，还么有看到js的异步和同步处理，只是在网上看到这个解释。用for循环遍历的话，每次取出的值是不正常的，图片只能读取到最后一个
        console.log("遍历",i, file);
        // files的索引是0开始 但是表格中已经右totalitemNum个图片 因此索引加上
        i = i + totalitemNum;
        // imghtml参考的是上面提到的博客的代码  
        var imghtml = '<div id="uploadList_'+ i +'" class="upload_append_list"><p><strong>' + file.name.substring(0,10) + '</strong>&nbsp'+ 
        '<a href="javascript:" class="upload_delete" title="删除" data-index="'+ i +'">删除</a><br />' +
        '<img name = "docfiless" id="uploadImage_' + i +'" class="upload_image" height="256" width="450"/></p>'+
        '<span id="uploadProgress_' + i + '" class="upload_progress">'+'<p>Analyse</p><span></span>'+'<span></span>'+'<span></span>'+'</span>' +
    '</div>'+'<div id="result" class="upload_result"></div>';
        // 根据图片索引的特殊 可以看到偶数都是在第一列 奇数序号图片都是在第二列 如果是偶数 需要给表格新建一行存放图片
        if(i%2==0){
            console.log("新建一行",i);
            let newRow = tb.insertRow(i/2); // 序号i的图片对应的是i/2行图片
            let newcell = newRow.insertCell(0); // 然后在新建的这一行，新建第一列
            newcell.innerHTML = imghtml;  // 那么该行该列的内容就是上面的imghtml
            newcell.style.align = "center";
        }
        else{
            console.log("在原始行添加",i); // 奇数直接在它对应的那一行新加一列即可
            let y = tb.rows[Math.floor(i/2)].insertCell(1);
            y.innerHTML = imghtml;
            y.style.align="center";
        }

        // 上面完成了，但是可以看到imghtml中的图片 还没有定义它的src属性，所以还需要设置src属性
        var imgid = document.getElementById("uploadImage_"+i);  // 取出对应的图片标签

        var re = new FileReader();
        re.onload = function(re) {
            console.log("imgid获取序号",i);  // 用for循环遍历files的话 可以看到这里的i始终将是最后一个 就出现问题
            if(imgid){ 
                imgid.src = re.target.result;  // re.target.result就是读出的文件base64编码字符串
            }
        }
        re.readAsDataURL(file);

        // imghtml中也添加了一个删除操作  因此对它绑定一个删除函数 函数参数是i 表示要删除的图片的序号
        $("#uploadList_"+i+ "> p > a").click(function(){
            deleteItem(i); // 传入的参数是索引 从0开始
        });
    });
}


// 获取当前表格中的图片数量
function getCurItemNums(){
    var tb = document.getElementById("imgViewtbody");
    var rowsNum = tb.rows.length; // 已经表格存在的行数
    var totalitemNum = rowsNum *2; // 总的图片数量 这个值是假设每一行都是两张图的情况下
    if(rowsNum >0){
        if(tb.rows[rowsNum-1].cells.length == 1){
            totalitemNum = totalitemNum -1; // 最后一行只有一张图
        }
    }
    return totalitemNum;// 当前的图像数量
}

// 删除索引为index的图片   分析：删除索引index的图片后，那么索引在它后面的图 应该往前移动一个格子 即它后面的图 ，原先在第一列，应该移动到上一行的第二列  原先在第二列，应该移动到同一行的第一列
// 同时 表格中的原先最后一个图片的位置 应该空出来了
function deleteItem(index){
    console.log("删除第",index,"个");  // 序号从0开始  index对应的是第

    var tb = document.getElementById("imgViewtbody");
    var curnums = getCurItemNums(); // 当前的图像数量

    if(curnums == index+1){ // 如果删除的本身就是最后一个元素 直接删除即可 不会影响到表格中的其他元素
        var row = Math.floor(index/2) ;// 第row行
        var cell = index%2; // 第cell列   索引index图片 对应的行和列是row和cell
        tb.rows[row].deleteCell(cell); // 直接删除它就可以了  // 如果这一行本身只有一个元素的话 执行后 会变成一个空的<tr></tr>， 下一次对其他行列的图片删除的时候 getCurItemNums函数的结果返回就不对
        if(tb.rows[row].cells.length == 0){
            tb.deleteRow(row); // 当前列没有元素的话，再把当前行删除了。这样才是完整的删除 不会有空的<tr>标签在
        }
    }
    else{
        // 如果不是最后一个元素 那么从index开始 ，依次用index+1的图片 替换掉当前index位置的图片即可
        for(var i=index;i<curnums-1;i++){
            // 让索引i的图像变为 i+1的图像  遍历到curnums-1即可 因为最后一张图后面没有了，直接删除最后一张即可
            console.log("修改图像",i)
            let row = Math.floor(i/2) ;// 第row行
            let cell = i%2; // 第cell列
    
            var imgid = $("#uploadImage_"+i)[0]; //取出索引i这个图片的标签 修改它的src 
            imgid.src = $("#uploadImage_"+(i+1)).attr("src"); // 新的src是它下一个的src

            // 还要修改每个表格中显示的文件名 #uploadList_2 > p > strong
            var filename = $("#uploadList_"+i+ "> p > strong")[0];
            filename.innerHTML = $("#uploadList_"+(i+1)+ "> p > strong").html();
        }

        // 最后删除最后一个元素  同样需要判断最后一行是不是空的tr然后删除行
        var row = Math.floor((curnums-1)/2) ;// 第row行
        var cell = (curnums-1)%2; // 第cell列
        tb.rows[row].deleteCell(cell); // 直接删除它就可以了
        if(tb.rows[row].cells.length == 0){
            tb.deleteRow(row); 
        }
    }
}
>>>>>>> 3efa523ff0c2328ea326da8c6ab7d5be45143578
