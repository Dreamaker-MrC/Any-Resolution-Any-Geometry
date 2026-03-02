document.addEventListener('DOMContentLoaded', domReady);

        function domReady() {
            new Dics({
                container: document.querySelectorAll('.b-dics')[0],
                hideTexts: false,
                textPosition: "bottom"

            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[1],
                hideTexts: false,
                textPosition: "bottom"
            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[2],
                hideTexts: false,
                textPosition: "bottom"
            });
            new Dics({
                container: document.querySelectorAll('.b-dics')[3],
                hideTexts: false,
                textPosition: "bottom"
            });
        }

        function objectSceneEvent(idx) {
            let dics = document.querySelectorAll('.b-dics')[0]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'images/zs1';
                        break;
                    case 1:
                        image.src = 'images/zs2';
                        break;
                    case 2:
                        image.src = 'images/zs3';
                        break;
                    // case 3:
                    //     image.src = 'images/zs4';
                    //     break;
                    // case 4:
                    //     image.src = 'images/zs5';
                    //     break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_pred.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_base.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("object-scale-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        function id_cs(idx) {
            let dics = document.querySelectorAll('.b-dics')[2]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'images/cs_id1';
                        break;
                    case 1:
                        image.src = 'images/cs_id2';
                        break;
                    case 2:
                        image.src = 'images/cs_id3';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_zoe.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("id-cs-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }


        function id_kitti(idx) {
            let dics = document.querySelectorAll('.b-dics')[1]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'images/kitti_id1';
                        break;
                    case 1:
                        image.src = 'images/kitti_id2';
                        break;
                    case 2:
                        image.src = 'images/kitti_id3';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_zoe.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("id-kitti-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }

        
        function id_scannet(idx) {
            let dics = document.querySelectorAll('.b-dics')[3]
            let sections = dics.getElementsByClassName('b-dics__section')
            let imagesLength = 3
            for (let i = 0; i < imagesLength; i++) {
                let image = sections[i].getElementsByClassName('b-dics__image-container')[0].getElementsByClassName('b-dics__image')[0]
                switch (idx) {
                    case 0:
                        image.src = 'images/scan_id1';
                        break;
                    case 1:
                        image.src = 'images/scan_id2';
                        break;
                    case 2:
                        image.src = 'images/scan_id3';
                        break;
                }
                switch (i) {
                    case 0:
                        image.src = image.src + '.jpg';
                        break;
                    case 1:
                        image.src = image.src + '_ours.jpg';
                        break;
                    case 2:
                        image.src = image.src + '_zoe.jpg';
                        break;
                }
            }

            let scene_list = document.getElementById("id-scannet-recon").children;
            for (let i = 0; i < scene_list.length; i++) {
                if (idx == i) {
                    scene_list[i].children[0].className = "nav-link active"
                }
                else {
                    scene_list[i].children[0].className = "nav-link"
                }
            }
        }