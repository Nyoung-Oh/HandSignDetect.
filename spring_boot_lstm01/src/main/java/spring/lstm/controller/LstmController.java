package spring.lstm.controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.servlet.ModelAndView;
import org.json.*;
import java.util.*;
import org.springframework.stereotype.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.servlet.*;
import org.apache.hc.client5.http.classic.methods.*;
import org.apache.hc.client5.http.entity.*;
import org.apache.hc.client5.http.entity.mime.*;
import org.apache.hc.client5.http.impl.classic.*;
import org.apache.hc.core5.http.*;
import org.apache.hc.core5.http.io.entity.*;
import org.apache.hc.core5.http.message.*;
import java.nio.charset.*;
import jakarta.servlet.http.*;

@Controller
public class LstmController {

	// url : webcamControl.do 일때 실행
	// method=RequestMethod.GET
	@RequestMapping(value="/webcamControl.do",
						   method=RequestMethod.GET)
	
	public ModelAndView inserBookForm() {
		ModelAndView mav = new ModelAndView();
		// webcam.html로 이동할 객체
		mav.setViewName("webcam.html");
		return mav;
	}
	
	// 컨트롤러가 Html 또는 JSP를 refresh 하거나
	// 다른 Html JSP로 이동하지 않고 결과를 리턴함
	@ResponseBody
	// url : sendLstm.do 실행할 url
	// method = RequestMethod.POST : Post 방식의 요청만 실행할 것임
	@RequestMapping(value = "/sendLstm.do",
					method = RequestMethod.POST)
	
	public String SendImageToRestServer(
		@RequestBody String data // 웹캠 화면이 저장될 매개 변수
			) throws Exception {
		// 웹캠 화면 저장된 data에서 data : 문자열 삭제 (웹캠 화면에 data : 문자열 있음)
		data = data.replace("data:", "");
		System.out.println("data(after replace)=" + data);
		
		// new JSONObject(data) : 웹캠 화면에서 저장된 데이터를 JSONObject 객체로 변환
		JSONObject dataObject = new JSONObject(data);
		System.out.println("dataObject=" + dataObject);
		
		// dataObject에서 key img_data의 value 리턴
		JSONArray cam_data_arr = ((JSONArray) dataObject.get("img_data"));
		System.out.println("cam_data_arr=" + cam_data_arr);
		
		// RestServer로 전송할 Json객체 restSendData 생성
		JSONObject restSendData = new JSONObject();
		restSendData.put("data", cam_data_arr);
		System.out.println("restSendData=" + restSendData);
		
		// 접속 할 Rest 서버 url 설정
		HttpPost httpPost = new HttpPost("http://localhost:5000/lstm_detect");
		// 전송할 데이터 타입 설정
		httpPost.addHeader("Content-Type", "application/json;charset=utf-8");
		// 리턴 받을 데이터 타입 설정
		httpPost.setHeader("Accept", "application/json;charset=utf-8");
		
		// 캠 화면 이미지 저장
		StringEntity stringEntity = new StringEntity(restSendData.toString());
		
		// Rest Server로 전송할 객체 생성
		httpPost.setEntity(stringEntity);
		
		// Rest 서버의 함수를 호출할 객체 생성
		CloseableHttpClient httpclient = HttpClients.createDefault();
		// Rest 서버의 함수를 호출하고 리턴 값을 가져올 객체 생성
		CloseableHttpResponse response2 = httpclient.execute(httpPost);
		
		// response2.getEntity() : Rest Server의 리턴 값을 가져옴
		// Charset.forName("UTF-8") : Rest Server의 리턴 값을 UTF-8로 인코딩
		// EntityUtils.toString() : Rest Server의 리턴 값을 String으로 변환해서 lstm_message 변수에 저장
		String lstm_message = EntityUtils.toString(response2.getEntity(), Charset.forName("UTF-8"));
		
		// System.out.println("lstm_message = " + lstm_message);
		
		return lstm_message;
		
	}
}
