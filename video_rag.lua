--[[ 

Copyright (C) 2024 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom
the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
OR OTHER DEALINGS IN THE SOFTWARE.

SPDX-License-Identifier: MIT ]]



text = nil
uri = nil

function descriptor()
    return {
        title = "Video Retrieval",
        version = "0.1",
        author = "Your Name",
        capabilities = {}
    }
end

function activate()
    vlc.msg.info("Video Retrieval")
    dlg = vlc.dialog("Video Retrieval")
    button_play = dlg:add_button("Search", click_play, 1, 8, 8, 1)
    video_path_label = dlg:add_label("<b>Video Path</b>", 1, 1, 4, 1)
	-- Optional: modify your videos path below. If not modify it from the VLC extension GUI 
    label = dlg:add_text_input("C:\\Users\\LNL\\Videos\\Source_Video", 1, 2, 10, 1)
    input_label = dlg:add_label("<b>Prompt</b>",1, 3, 4, 1)
    input = dlg:add_text_input(text,1, 4, 10, 1)
    dlg:show()    

end

function deactivate()
    vlc.msg.info("Video Rag deactivated")
    --stop_periodic_task()

    -- Clean up resources here
end



function test_server(uri,message)
    vlc.msg.info("In test server")
    -- Modify this line to be the same as line 10 of client_RAG.py
    local script_path = debug.getinfo(1, "S").source:sub(2)
    local script_dir = string.match(script_path, "(.*\\)")
    local python_script = script_dir .."client_RAG.py "
    local final_path = string.gsub(python_script, '\\','/')

    local user_data_dir = vlc.config.userdatadir()
    local file = user_data_dir .. "\\cache_test.txt" 
    
    os.remove(file)
    vlc.msg.info(file)
	-- Modify this line to be the path to client_RAG.py
    local command =string.format('pythonw "%s" "%s" "%s"',final_path, uri,message)
    
	vlc.msg.info(command)

    local handle = io.popen(command)  
    
    vlc.keep_alive()
    while true do
        vlc.keep_alive()
        --vlc.msg.info("In keep alive")
        local f = io.open(file, "rb")
        if f then 
            vlc.msg.info("Got result")
            f:close()
            break
        end
       
    end
 
    local result = handle:read("*a")
    
    handle:close()


-- local result = os.execute(command)
    if result then
        vlc.msg.info("python script executed" .. result)
    else
        vlc.msg.err("python script Failed")
    end 
    return result
end

function video_path()
    local media = vlc.input.item()
    --local player = vlc.object.input()
    
    -- Check if media exists
    if media then
        -- Get the path of the media
        local media_path = vlc.strings.decode_uri(media:uri())
        vlc.msg.info("Playing item directly path: " .. media:uri())
        uri = string.gsub(media_path, '^file:///','')
        uri = string.gsub(uri, '/','\\')
        -- Display the path
        vlc.msg.info("Playing item path: " .. uri)
        --local time = vlc.input.get_position()
        

        vlc.msg.info("Playing item status 1" .. vlc.playlist.status())
        if vlc.playlist.status() == "playing" then
            vlc.playlist.pause()
        end
        vlc.msg.info("Playing item status 2" .. vlc.playlist.status())
        
    end
        
    return uri
       
        
    
    end




function seek_by_frame(frame_number)
    --local frame_number = 1950 --tonumber(frame_input:get_text())
    if not frame_number then
        vlc.msg.warn("Invalid time!")
        return
    end

    local item = vlc.input.item()
    if not item then
        vlc.msg.warn("No video currently playing!")
        return
    end

    local player = vlc.object.input()
    --local time_in_seconds = frame_number / fps
    local time_in_seconds = frame_number
    vlc.msg.info("time_in_seconds" .. time_in_seconds .. " seconds)") 


    --local tms = 160
    vlc.var.set(player,"time",tonumber(time_in_seconds)*1000000) --String2time(time_in_seconds))
   
    vlc.playlist.play()
    --end

    vlc.msg.info("Seeking to frame " .. frame_number .. " (" .. time_in_seconds .. " seconds)")
end




function video_by_path(video_file)
	
    if video_file and video_file ~= "" then
         --vlc.strings.make_uri( video_file ) 
       -- local url = "file:///d:/VisiualRag/vlc_video_rag/Happyness/video/Happyness_32_160_170.mp4" 

        
        video_file = string.gsub(video_file, "[\r\n]", "")
		local v = {}
		local i = 1
		sep = "%s"
		for str in string.gmatch(video_file, "([^"..sep.."]+)") do
			v[i] = str
			i = i + 1
		end
		

		
		frame_number = v[2]
		
		vlc.msg.info("Timestamp: " .. frame_number)
        vlc.playlist.clear()
		vlc.playlist.add({{ path = v[1] }})
		vlc.playlist.play()
		
		vlc.msg.info("Playing video: " .. v[1])
		local item = vlc.input.item()
		
		while true do
			local item = vlc.input.item()
			if item then
				break
			end
       
       
		end

		
		local player = vlc.object.input()

		
		local time_in_seconds = frame_number
		 


		--local tms = 160
		vlc.var.set(player,"time",tonumber(time_in_seconds)*1000000) --String2time(time_in_seconds))
		vlc.msg.info("time_in_seconds" .. time_in_seconds .. " seconds)")
	   
		--vlc.playlist.play()
		--end

		vlc.msg.info("Seeking to frame " .. frame_number .. " (" .. time_in_seconds .. " seconds)")		
		
		--seek_by_frame(v[2])
        --vlc.playlist.play()
    else
        vlc.msg.warn("No path specified")
    end
end




function click_play()

    local media = vlc.input.item()

    if media then
        uri = video_path()
        dlg:del_widget(label)
        label = dlg:add_text_input(uri, 1, 2, 10, 1)
        dlg:update()

    else
        local video_folder = label:get_text()
        uri = video_folder
    end
        




    local message = input:get_text() --"Hello LUA"

    
    local response = test_server(uri,message) --send_to_python_server(message)
    vlc.msg.info("Python Server Response: " .. response)

    if media then   

        seek_by_frame(response)
    else
        video_by_path(response)
    end
  

end
