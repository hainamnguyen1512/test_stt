 var recordData

      var fileTotalSeconds = 0;

      var recordTotalSeconds = 0;

      function getResTable(type, wavenet, lm, google) {
        $(`#${type}-time`).html('')
        $(`#${type}-progress-bar`).html('')

        $(`#${type}-res-table`).html(`
          <div><strong>Elapsed time</strong>: ${pad(parseInt(type == 'file' ? getFileTime()/60 : getRecordTime()/60))}:${pad(type == 'file' ? getFileTime()%60 : getRecordTime()%60)}</div>
          <div>
            <table class="table table-bordered table-striped table-responsive table-hover" style="margin: 10px auto; width: 80%; text-align: center">
              <thead style="background-color: #32383e">
                <tr style="color: white">
                  <th style="text-align: center" colspan="2">Result</th>
                </tr>
              </thead>

              <tbody>
                <tr><td style="width: 50%"><strong>Wavenet result</strong></td><td>${wavenet}</td></tr>
                <tr><td style="width: 50%"><strong>Corrected by Language Model</strong></td><td>${lm}</td></tr>
                <tr><td style="width: 50%"><strong>Google transcript</strong></td><td>${google}</td></tr>
              </tbody>
            </table>
          </div>
        `)
      }

      function getError(type) {
        $(`#${type}-time`).html('')
        $(`#${type}-progress-bar`).html('')

        $(`#${type}-res-table`).html(`Error!`)
      }

      function runMain(type, formdata) {
        $(`#${type}-time`).html(`
          <label id="${type}-minutes">00</label>:<label id="${type}-seconds">00</label>
        `)

        $(`#${type}-upload`).html(``)
        $(`#${type}-recognize`).html(``)
        $(`#${type}-correct`).html(``)
        $(`#${type}-google`).html(``)
        $(`#${type}-res-table`).html(``)

        $(`#${type}-progress-bar`).html(`
          <div class="progress">
            <div
              class="progress-bar progress-bar-striped active"
              role="progressbar"
              aria-valuenow="100"
              aria-valuemin="0"
              aria-valuemax="100"
              style="width: 100%"
            ></div>
          </div>
        `)

        var minutesLabel = document.getElementById(`${type}-minutes`);
        var secondsLabel = document.getElementById(`${type}-seconds`);
        type == 'file' ?
        fileInterval = setInterval(setFileTime, 1000) :
        recordInterval = setInterval(setRecordTime, 1000)

        function setFileTime() {
          ++fileTotalSeconds;
          secondsLabel.innerHTML = pad(fileTotalSeconds % 60);
          minutesLabel.innerHTML = pad(parseInt(fileTotalSeconds / 60));
        }

        function setRecordTime() {
          ++recordTotalSeconds;
          secondsLabel.innerHTML = pad(recordTotalSeconds % 60);
          minutesLabel.innerHTML = pad(parseInt(recordTotalSeconds / 60));
        }

        $(`#${type}-upload`).html('Uploading ... ')

        $.ajax({
          url: `/upload${type}`,
          type: "post",
          dataType: "html",
          data: formdata,
          encode: true,
          processData: false,
          cache: false,
          contentType: false,

          success: function( data ){
 0 0 0           console.log('Uploaded!')
            console.log(data)
            $(`#${type}-upload`).html(`Uploading ... <strong>DONE!</strong> (${data} s)`)
            $(`#${type}-recognize`).html('Recognizing ... ')

            $.ajax({
              url: `/wavenet${type}`,
              type: "post",
              dataType: "html",
              data: '',

              success: function( data ){
                data = JSON.parse(data)
                console.log('Recognized!')
                console.log(data.result[0])
                console.log(data.result[1])

                type == 'file' ? file_wavenet = data.result[0] : record_wavenet = data.result[0]

                $(`#${type}-recognize`).html(`Recognizing ... <strong>DONE!</strong> (${data.result[1]} s)`)
                $(`#${type}-correct`).html('Correcting ... ')

                $.ajax({
                  url: `/lm${type}`,
                  type: "post",
                  dataType: "html",
                  data: '',

                  success: function( data ){
                    data = JSON.parse(data)
                    console.log('Corrected!')
                    console.log(data.result[0])
                    console.log(data.result[1])

                    type == 'file' ? file_lm = data.result[0] : record_lm = data.result[0]

                    $(`#${type}-correct`).html(`Correcting ... <strong>DONE!</strong> (${data.result[1]} s)`)
                    $(`#${type}-google`).html('Getting Google transcript ... ')

                    $.ajax({
                      url: `/google${type}`,
                      type: "post",
                      dataType: "html",
                      data: '',

                      success: function( data ){
                        data = JSON.parse(data)
                        console.log('Got Google transcript!')
                        console.log(data.result[0])
                        console.log(data.result[1])

                        type == 'file' ? file_google = data.result[0] : record_google = data.result[0]

                        $(`#${type}-google`).html(`Getting Google transcript ... <strong>DONE!</strong> (${data.result[1]} s)`)

                        getResTable(type,
                                    type == 'file' ? file_wavenet : record_wavenet,
                                    type == 'file' ? file_lm : record_lm,
                                    type == 'file' ? file_google : record_google)

                        clearInterval(type == 'file' ? fileInterval : recordInterval)
                        type == 'file' ? fileTotalSeconds = 0 : recordTotalSeconds = 0
                      },
                      error: function(error) {
                        console.log(error)
                        getError(type)
                      }
                    })
                  },
                  error: function(error) {
                    console.log(error)
                    getError(type)
                  }
                })
              },
              error: function(error) {
                console.log(error)
                getError(type)
              }
            })
          },
          error: function(error) {
            console.log(error)
            getError(type)
          }
        })
      }

      function getFileTime() {
        return fileTotalSeconds
      }

      function getRecordTime() {
        return recordTotalSeconds
      }

      function pad(val) {
        var valString = val + "";
        if (valString.length < 2) {
          return "0" + valString;
        } else {
          return valString;
        }
      }

      $('#record-btn').click(function() {
        $( this ).css( "background-color", "white" )
        $('#record-icon').css('color', 'red')
        $('#stop-icon').css('color', '#333')
      })

      $('#stop-btn').click(function() {
        $( this ).css( "background-color", "white" )
        $('#stop-icon').css('color', 'red')
        $('#record-icon').css('color', '#333')
      })

      $('#record-btn').mouseenter(function() {
        $( this ).css( "background-color", "#e6e6e6" )
      }).mouseleave(function() {
        $( this ).css( "background-color", "white" )
      })

      $('#stop-btn').mouseenter(function() {
        $( this ).css( "background-color", "#e6e6e6" )
      }).mouseleave(function() {
        $( this ).css( "background-color", "white" )
      })

      $('#input-file').change(function() {
        var input = document.getElementById("input-file")

        file = input.files[0]
        console.log(file)

        var sound = document.getElementById("file-sound")
        var reader = new FileReader()
        reader.onload = function(e) {
          sound.src = this.result
          sound.controls = true
        }
        reader.readAsDataURL(file)
      })

      $("#recognize-file").submit(function( event ) {
        event.preventDefault()

        var formdata = new FormData()
        formdata.append("file", file)

        runMain('file', formdata)
      })

      navigator.getUserMedia = (  navigator.getUserMedia ||
                                  navigator.webkitGetUserMedia ||
                                  navigator.mozGetUserMedia ||
                                  navigator.msGetUserMedia )

      var record = document.querySelector('#record-btn')
      var stop = document.querySelector('#stop-btn')
      var audioSection = $('#record-sound-section')
      var canvas = document.querySelector('#visualizer')

      var audioCtx = new (window.AudioContext || webkitAudioContext)();
      var canvasCtx = canvas.getContext("2d")

      if (navigator.getUserMedia) {
        console.log('getUserMedia supported.');
        navigator.getUserMedia (
          // constraints - only audio needed for this app
          {
            audio: true
          },

          // Success callback
          function(stream) {
            var mediaRecorder = new MediaRecorder(stream);

            visualize(stream);

            record.onclick = function() {
              mediaRecorder.start();
              console.log(mediaRecorder.state);
              console.log("recorder started");
              audioSection.html('Recording...')
            }

            stop.onclick = function() {
              mediaRecorder.stop();
              console.log(mediaRecorder.state);
              console.log("recorder stopped");
            }

            mediaRecorder.ondataavailable = function(e) {
              console.log("data available");

              var clipName = prompt('Enter a name for your sound clip');

              var audioURL = window.URL.createObjectURL(e.data);

              recordData = e.data

              console.log(audioURL)

              audioSection.html(`
              <div class="input-group" id="record-sound-group">
                <span class="input-group-addon">${clipName || 'recorded'}</span>
                <audio id="record-sound" name="record-sound" control></audio>
                <span class="input-group-btn">
                  <button class="btn btn-default" type="submit">Submit</button>
                </span>
              </div>
              `);

              var sound = document.getElementById("record-sound")
              sound.src = audioURL
              sound.controls = true
            }
          },

          // Error callback
          function(err) {
            console.log('The following gUM error occured: ' + err);
          }
        );
      } else {
        console.log('getUserMedia not supported on your browser!');
      }

      function visualize(stream) {
        var source = audioCtx.createMediaStreamSource(stream);

        var analyser = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        var bufferLength = analyser.frequencyBinCount;
        var dataArray = new Uint8Array(bufferLength);

        source.connect(analyser);
        //analyser.connect(audioCtx.destination);

        WIDTH = canvas.width
        HEIGHT = canvas.height;

        draw()

        function draw() {

          requestAnimationFrame(draw);

          analyser.getByteTimeDomainData(dataArray);

          canvasCtx.fillStyle = 'rgb(200, 200, 200)';
          canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

          canvasCtx.lineWidth = 1.5;
          canvasCtx.strokeStyle = 'rgb(0, 0, 0)';

          canvasCtx.beginPath();

          var sliceWidth = WIDTH * 1.0 / bufferLength;
          var x = 0;


          for(var i = 0; i < bufferLength; i++) {

            var v = dataArray[i] / 128.0;
            var y = v * HEIGHT/2;

            if(i === 0) {
              canvasCtx.moveTo(x, y);
            } else {
              canvasCtx.lineTo(x, y);
            }

            x += sliceWidth;
          }

          canvasCtx.lineTo(canvas.width, canvas.height/2);
          canvasCtx.stroke();
        }
      }

      $("#recognize-record").submit(function( event ) {
        event.preventDefault()

        var formdata = new FormData()
        formdata.append('file', recordData)

        runMain('record', formdata)
      })