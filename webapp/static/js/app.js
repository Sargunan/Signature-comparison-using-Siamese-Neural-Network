$(document).ready(function(){
    console.log('test')
    var $status = $('.status');
    var results = {
        bboxes: []
    };

    var drawBBox = function() {
        var threshold = parseInt($("#bbox-slider").text());
        var c = document.getElementById("img-canvas");
        var ctx = c.getContext("2d");
        var img = document.getElementById("loaded-img");

        let height = 500;
        let width = 500;
        ctx.clearRect(0, 0, width, height);
        ctx.drawImage(img,0,0, width, height);

        results.bboxes.forEach(function(d) {
            var x1, y1, x2, y2;
            if (d.score > threshold) {
                ctx.beginPath();
                x1 = parseInt(d.x1 * width);
                y1 = parseInt(d.y1 * height);
                x2 = parseInt(d.x2 * width);
                y2 = parseInt(d.y2 * height);
                b_w = x2-x1
                b_h = y2-y1
                ctx.rect(x1, y1, b_w, b_h);
                ctx.lineWidth = 3;
                ctx.strokeStyle = 'white';
                ctx.stroke();

            }
        });
    };

    $('#img').change(function(event) {
        var obj = $(this)[0];
        // console.log(obj.files, obj.files[0])
        $status.html('');

        if (obj.files && obj.files[0]) {
            var fileReader = new FileReader();
            fileReader.onload = function(event) {
                $('.img-hidden').html(
                    `<img id='loaded-img' src='${event.target.result}'/>`
                );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                ctx.drawImage(img,0,0, 500,500);
                });
            }
            fileReader.readAsDataURL(obj.files[0]);
        }
    });

    $('#predict').submit(function(event) {
        event.preventDefault();

        if ($('#img')[0].files.length === 0) {
            return false;
        }

        var imageData = new FormData($(this)[0]);
        console.log($(this)[0]);
        $status.html(
            `<span class='eval'>Evaluating...</span>`
        );
        
        $.ajax({
            url: '/predict',
            type: 'POST',
            processData: false,
            contentType: false,
            dataType: 'json',
            data: imageData,

            success: function(responseData) {
                if (responseData.error === 'bad-type') {
                    $status.html(
                        `<span class='eval'>Valid file types are .jpg and .png</span>`
                    );
                } else {
                    results["bboxes"] = responseData["bboxes"];
                    let preData = JSON.stringify(responseData, null, '\t');
                    $('#productImg img').src = 'D:\\cocoapp-master\\cocoapp-master\\cocoapp\\static\\images\\AxisAIChallenge.png';
                    $status.html(
                               `<span class='result success'>Results</span>
                                 <pre>${preData}</pre>`
                            );
                    // Draw Bounding boxes
                    //drawBBox();
                 }
            },
            error: function() {
                $status.html(
                    `<span class='eval'>Something went wrong, try again later.</span>`
                );
            }
        });
    });
    
    
    $('#upload').submit(function(event) {
        event.preventDefault();

        if ($('#img')[0].files.length === 0) {
            return false;
        }

        var imageData = new FormData($(this)[0]);
        console.log($(this)[0]);
        $status.html(
            `<span class='eval'>Uploading...</span>`
        );
        
        $.ajax({
            url: '/upload',
            type: 'POST',
            processData: false,
            contentType: false,
            dataType: 'json',
            data: imageData,

            success: function(responseData) {
                if (responseData.error === 'bad-type') {
                    $status.html(
                        `<span class='eval'>Valid file types are .jpg and .png</span>`
                    );
                } else {
                    results["bboxes"] = responseData["bboxes"];
                    let preData = JSON.stringify(responseData, null, '\t');
                    $('#productImg img').src = 'D:\\cocoapp-master\\cocoapp-master\\cocoapp\\static\\images\\AxisAIChallenge.png';
                    $status.html(
                               `<span class='result success'>Results</span>
                                 <pre>${preData}</pre>`
                            );
                    // Draw Bounding boxes
                    //drawBBox();
                 }
            },
            error: function() {
                $status.html(
                    `<span class='eval'>Something went wrong, try again later.</span>`
                );
            }
        });
    });

    $(".sample_img").click(function() {
         
        // add active class to clicked picture
        $(".sample_img").removeClass("active");
        $(this).addClass("active");

     
        // grab image url
        var url = $(this).attr("src");

        // read url into blob using XHR
        var request = new XMLHttpRequest();
        request.open('GET', url, true);
        request.responseType = 'blob';
        
        request.onload = function() {
            var reader = new FileReader();
            reader.readAsDataURL(request.response);

            // draw canvas of selected sample image
            reader.onload =  function(e){
                // console.log('DataURL:', e.target.result);
                $('.img-hidden').html(
                    `<img id='loaded-img' src='${e.target.result}'/>`
                    );
                var c = document.getElementById("img-canvas");
                var ctx = c.getContext("2d");
                var img = document.getElementById("loaded-img");
                img.addEventListener("load", function(e) {
                    ctx.drawImage(img,0,0, 500, 500);
                });

                // blob into form data
                var blob = request.response;
                var fd = new FormData();
                fd.set('file', blob);
                // console.log(formD);

                $.ajax({
                    url: '/predict',
                    type: 'POST',
                    processData: false,
                    contentType: false,
                    dataType: 'json',
                    data: fd,

                    success: function(responseData) {
                        if (responseData.error === 'bad-type') {
                            console.log('no good')
                            $status.html(
                                `<span class='eval'>Valid file types are .jpg and .png</span>`
                            );
                        } else {
                            results["bboxes"] = responseData["bboxes"];
                            let preData = JSON.stringify(responseData, null, '\t');
                            $status.html(
                                `<span class='result success'>Results</span>
                                 <pre>${preData}</pre>`
                            );
                            // Draw Bounding boxes
                            drawBBox();
                         }
                    },
                    error: function() {
                        $status.html(
                            `<span class='eval'>Something went wrong, try again later.</span>`
                        );
                    }
                });
            };
        };
        request.send();
    });   
    
    $('#bbox-slider').slider({
        formatter: function(value) {
            return value;
        }
    })
    .on('slide', drawBBox)
    .data('slider');
});